import _pickle as pickle
import os
from multiprocessing import Process, Queue, Lock, Manager, Value, Lock
import queue
import torch
from tensorboardX import SummaryWriter
import numpy as np
import multiprocessing as mp

import utils
from memory import BatchStorage
from wrapper import make_atari, wrap_atari_dqn
from model import DuelingDQN
from arguments import argparser
import time
import zlib
import grpc
import apex_data_pb2, apex_data_pb2_grpc
from concurrent.futures import ThreadPoolExecutor

def get_environ():
    actor_ip = os.environ.get('ACTOR_IP', '-1')
    actor_id = int(os.environ.get('ACTOR_ID', '-1'))
    n_actors = int(os.environ.get('N_ACTORS', '-1'))
    replay_ip = os.environ.get('REPLAY_IP', '-1')
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    registerActorPort = os.environ.get('REGISTERACTORPORT', '-1')
    sendBatchPrioriPort = os.environ.get('SENDBATCHPRIORIPORT', '-1')
    updatePrioriPort = os.environ.get('UPDATEPRIORIPORT', '-1')
    sampleDataPort = os.environ.get('SAMPLEDATAPORT', '-1')
    sampleDataPortReplay = os.environ.get('SAMPLEDATAPORTRPLAY', '-1')
    parameterPort = os.environ.get('PARAMETERPORT', '-1')
    return actor_ip, actor_id, n_actors, replay_ip, learner_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort, sampleDataPortReplay, parameterPort


def recv_param(param_queue, port_dict, req_param_queue):
    """
    initial connect learner server recv parameters
    """
    MAX_MESSAGE_LENGTH = 100*1024*1024
    conn = grpc.insecure_channel(port_dict['learner_ip'] + ':' + port_dict['parameterPort'],options = [('grpc.max_send_message_length',MAX_MESSAGE_LENGTH),('grpc.max_receive_message_length',MAX_MESSAGE_LENGTH)])
    client = apex_data_pb2_grpc.SendParameterStub(channel=conn)
    while True:
        req_param_queue.get(block=True)
        param_dict = {}
        response = client.Send(apex_data_pb2.ParametersRequest(param_req=True))
        for p in response:
            shape = tuple(p.shape)
            param_dict[p.key] = torch.from_numpy(np.array(p.values).reshape(shape))
        param_queue.put(param_dict)


def exploration(args, actor_id, n_actors, param_queue, send_queue, req_param_queue):
    writer = SummaryWriter(comment="-{}-actor{}".format(args.env, actor_id))

    env = make_atari(args.env)
    env = wrap_atari_dqn(env, args)

    seed = args.seed + actor_id
    utils.set_global_seeds(seed, use_torch=True)
    env.seed(seed)

    model = DuelingDQN(env)
    epsilon = args.eps_base ** (1 + actor_id / (n_actors - 1) * args.eps_alpha)
    storage = BatchStorage(args.n_steps, args.gamma)
    req_param_queue.put(True)
    param = param_queue.get(block=True)
    model.load_state_dict(param)
    param = None
    print("Received First Parameter!")

    episode_reward, episode_length, episode_idx, actor_idx = 0, 0, 0, 0
    state = env.reset()
    while True:
        action, q_values = model.act(torch.FloatTensor(np.array(state)), epsilon)
        next_state, reward, done, _ = env.step(action)
        com_state = zlib.compress(np.array(state).tobytes())
        storage.add(com_state, reward, action, done, q_values)

        state = next_state
        episode_reward += reward
        episode_length += 1
        actor_idx += 1

        if done or episode_length == args.max_episode_length:
            state = env.reset()
            writer.add_scalar("actor/episode_reward", episode_reward, episode_idx)
            writer.add_scalar("actor/episode_length", episode_length, episode_idx)
            episode_reward = 0
            episode_length = 0
            episode_idx += 1

        if actor_idx % args.update_interval == 0:
            try:
                req_param_queue.put(True)
                param = param_queue.get(block=True)
                model.load_state_dict(param)
                print("Updated Parameter..")
            except queue.Empty:
                pass

        if len(storage) == args.send_interval:
            batch, prios = storage.make_batch()
            send_queue.put((batch, prios))
            batch, prios = None, None
            storage.reset()

def sendBatchPrioriProcess(send_queue, port_dict, local_buffer, next_idx, buffer_size, actor_id, lock):
    """
    initial connect replay server send batch priori
    """
    conn = grpc.insecure_channel(port_dict['replay_ip'] + ':' + port_dict['sendBatchPrioriPort'])
    client = apex_data_pb2_grpc.SendBatchPrioriStub(channel=conn)
    while True:
        batch, priori = send_queue.get(block=True)
        idx_send = []
        timestamp_send = []
        for sample in zip(*batch):
            timestamp = time.time()
            with lock:
                local_buffer[next_idx.value] = (sample, timestamp)
                idx_send.append(next_idx.value)
                timestamp_send.append(timestamp)
                next_idx.value = (next_idx.value+1)%buffer_size.value
        request = apex_data_pb2.BatchPrioriRequest(idxes = idx_send, prioris = priori.tolist(), actor_id = actor_id, timestamp=timestamp_send)
        response = client.Send(request)
        if response.response:
            print("Send Batch...")
        batch, priori = None, None

class SendRealData(apex_data_pb2_grpc.SendRealDataServicer):
    def Send(self, request, context):
        idxes = request.idxes
        cnt = 0
        for idx in idxes:
            with lock:
                (state, action, reward, next_state, done), timestamp = local_buffer[idx]
            data = apex_data_pb2.RealDataResponse()
            data.state = state
            data.action = action
            data.reward = reward
            data.next_state = next_state
            data.done = done
            data.idx = cnt
            data.timestamp = timestamp
            cnt += 1
            yield data


if __name__ == '__main__':
    mp.set_start_method('spawn')

    actor_ip, actor_id, n_actors, replay_ip, learner_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort, sampleDataPortReplay, parameterPort = get_environ()
    args = argparser()
    param_queue = Queue(maxsize=1)
    req_param_queue = Queue(maxsize=1)
    next_idx = Value("i", 0)
    buffer_size = Value("i", args.replay_buffer_size // n_actors)
    local_buffer = Manager().list(range(args.replay_buffer_size // n_actors))
    send_queue = Queue(maxsize=3)
    port_dict = Manager().dict()
    port_dict['replay_ip'] = replay_ip
    port_dict['learner_ip'] = learner_ip
    port_dict['registerActorPort'] = registerActorPort
    port_dict['sendBatchPrioriPort'] = sendBatchPrioriPort
    port_dict['updatePrioriPort'] = updatePrioriPort
    port_dict['sampleDataPort'] = sampleDataPort
    port_dict['sampleDataPortReplay'] = sampleDataPortReplay
    port_dict['parameterPort'] = parameterPort
    lock = Lock()
    """
    actor sends real data to replay buffer
    """
    sendRealDataServer = grpc.server(ThreadPoolExecutor(max_workers=8))
    apex_data_pb2_grpc.add_SendRealDataServicer_to_server(SendRealData(), sendRealDataServer)
    sendRealDataServer.add_insecure_port(actor_ip + ':' + sampleDataPortReplay)
    sendRealDataServer.start()

    #time.sleep(3)

    conn = grpc.insecure_channel(port_dict['learner_ip'] + ':' + port_dict['registerActorPort'])
    client = apex_data_pb2_grpc.RegisterActorStub(channel=conn)
    request = apex_data_pb2.ActorRegisterRequest(actor_ip=actor_ip, actor_id = actor_id, data_port = sampleDataPortReplay)
    response = client.Send(request=request)
    if response.response:
        print("actor {} regist success".format(actor_id))

    procs = [
        Process(target=exploration, args=(args, actor_id, n_actors, param_queue, send_queue, req_param_queue)),
        Process(target=recv_param, args=(param_queue, port_dict, req_param_queue)),
        Process(target=sendBatchPrioriProcess, args=(send_queue, port_dict, local_buffer, next_idx, buffer_size, actor_id, lock)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()
