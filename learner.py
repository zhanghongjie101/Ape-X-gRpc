"""
Module for learner in Ape-X.
"""
import time
import os
import _pickle as pickle
import threading
import queue

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Manager
from tensorboardX import SummaryWriter
import numpy as np
from wrapper import make_atari, wrap_atari_dqn

import utils
import wrapper
from model import DuelingDQN
from arguments import argparser
import time
import zlib
import grpc
import apex_data_pb2, apex_data_pb2_grpc
from concurrent.futures import ThreadPoolExecutor
import copy

def get_environ():
    n_actors = int(os.environ.get('N_ACTORS', '-1'))
    replay_ip = os.environ.get('REPLAY_IP', '-1')
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    registerActorPort = os.environ.get('REGISTERACTORPORT', '-1')
    sendBatchPrioriPort = os.environ.get('SENDBATCHPRIORIPORT', '-1')
    updatePrioriPort = os.environ.get('UPDATEPRIORIPORT', '-1')
    sampleDataPort = os.environ.get('SAMPLEDATAPORT', '-1')
    parameterPort = os.environ.get('PARAMETERPORT', '-1')
    return n_actors, replay_ip, learner_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort, parameterPort


def sample_batch(args, batch_queue, port_dict, device):
    """
    receive batch from replay and transfer batch from cpu to gpu
    """
    conn = grpc.insecure_channel(port_dict['replay_ip'] + ':' + port_dict['sampleDataPort'])
    client = apex_data_pb2_grpc.SampleDataStub(channel=conn)

    while True:
        res_batch = client.Send(apex_data_pb2.SampleDataRequest(batch_size=args.batch_size, beta = args.beta))
        batch_data = []
        batch_weights = []
        batch_idxes = []
        for res_data in res_batch:
            batch_data.append(res_data)
            batch_weights.append(res_data.weight)
            batch_idxes.append(res_data.idx)

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(len(batch_weights)):
            states.append(batch_data[i].state)
            actions.append(batch_data[i].action)
            rewards.append(batch_data[i].reward)
            next_states.append(batch_data[i].next_state)
            dones.append(batch_data[i].done)

        states = np.array([np.frombuffer(zlib.decompress(state), dtype=np.uint8).reshape((4,84,84)) for state in states])
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = np.array([np.frombuffer(zlib.decompress(state), dtype=np.uint8).reshape((4,84,84)) for state in next_states])
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        weights = torch.FloatTensor(batch_weights).to(device)

        batch = [states, actions, rewards, next_states, dones, weights, batch_idxes]
        #if batch_queue.full():
        #    print("batch_queue size of recv batch:{}".format(batch_queue.qsize()))
        batch_queue.put(batch)
        data, batch = None, None


def update_prios(prios_queue, port_dict):
    """
    initial connect replay server update priority
    """
    conn = grpc.insecure_channel(port_dict['replay_ip'] + ':' + port_dict['updatePrioriPort'])
    client = apex_data_pb2_grpc.UpdateBatchPrioriStub(channel=conn)
    while True:
        idxes, prios = prios_queue.get()
        response = client.Send(apex_data_pb2.UpdatePrioriRequest(idxes = idxes, prioris = prios))
        if response.response:
            pass


def train(args, n_actors, batch_queue, prios_queue, param_queue):
    """
    thread to fill parameter queue
    """
    def _fill_param():
        while True:
            model_dict = {}
            state_dict = model.state_dict()
            for k, v in state_dict.items():
                model_dict[k] = v.cpu().numpy()
            param_queue.put(model_dict)

    env = wrapper.make_atari(args.env)
    env = wrapper.wrap_atari_dqn(env, args)
    utils.set_global_seeds(args.seed, use_torch=True)

    model = DuelingDQN(env).to(args.device)
    tgt_model = DuelingDQN(env).to(args.device)
    tgt_model.load_state_dict(model.state_dict())

    writer = SummaryWriter(comment="-{}-learner".format(args.env))
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr, alpha=0.95, eps=1.5e-7, centered=True)
    model_dict = {}
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        model_dict[k] = v.cpu().numpy()
    param_queue.put(model_dict)
    threading.Thread(target=_fill_param).start()
    learn_idx = 0
    ts = time.time()
    while True:
        *batch, idxes = batch_queue.get()
        loss, prios = utils.compute_loss(model, tgt_model, batch, args.n_steps, args.gamma)
        grad_norm = utils.update_parameters(loss, model, optimizer, args.max_norm)
        prios_queue.put((idxes, prios))
        batch, idxes, prios = None, None, None
        learn_idx += 1

        if learn_idx % args.tensorboard_update_interval == 0:
            writer.add_scalar("learner/loss", loss, learn_idx)
            writer.add_scalar("learner/grad_norm", grad_norm, learn_idx)

        if learn_idx % args.target_update_interval == 0:
            print("Updating Target Network..")
            tgt_model.load_state_dict(model.state_dict())
        if learn_idx % args.save_interval == 0:
            print("Saving Model..")
            torch.save(model.state_dict(), "model.pth")
        if learn_idx % args.publish_param_interval == 0:
            param_queue.get()
        if learn_idx % args.bps_interval == 0:
            bps = args.bps_interval / (time.time() - ts)
            print("Step: {:8} / BPS: {:.2f}".format(learn_idx, bps))
            writer.add_scalar("learner/BPS", bps, learn_idx)
            ts = time.time()

class SendParameter(apex_data_pb2_grpc.SendParameterServicer):
    def Send(self, request, context):
        if request.param_req:
            param = param_queue.get()
            for k, v in param.items():
                value = v
                shape = list(value.shape)
                value = value.reshape(-1).tolist()
                single_param = apex_data_pb2.SingleParameter(key=k, values = value, shape=shape)
                yield single_param


if __name__ == '__main__':
    mp.set_start_method('spawn')
    """
    environment parameters
    """
    n_actors, replay_ip, learner_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort, parameterPort = get_environ()

    args = argparser()

    port_dict = Manager().dict()
    port_dict['replay_ip'] = replay_ip
    port_dict['learner_ip'] = learner_ip
    port_dict['registerActorPort'] = registerActorPort
    port_dict['sendBatchPrioriPort'] = sendBatchPrioriPort
    port_dict['updatePrioriPort'] = updatePrioriPort
    port_dict['sampleDataPort'] = sampleDataPort
    port_dict['parameterPort'] = parameterPort

    # TODO: Need to adjust the maxsize of prios, param queue
    batch_queue = Queue(maxsize=args.queue_size)
    prios_queue = Queue(maxsize=args.prios_queue_size)
    param_queue = Queue(maxsize=1)
    """
    send parameter from learner to actor
    """
    MAX_MESSAGE_LENGTH = 100*1024*1024
    sendParameterServer = grpc.server(ThreadPoolExecutor(max_workers=4),options = [('grpc.max_send_message_length',MAX_MESSAGE_LENGTH),('grpc.max_receive_message_length',MAX_MESSAGE_LENGTH)])
    apex_data_pb2_grpc.add_SendParameterServicer_to_server(SendParameter(), sendParameterServer)
    sendParameterServer.add_insecure_port(learner_ip + ':' + parameterPort)
    sendParameterServer.start()
    print("ready for parameter server")
    procs = [
        Process(target=train, args=(args, n_actors, batch_queue, prios_queue, param_queue)),
    ]

    for _ in range(args.n_send_prios_process):
        p = Process(target=update_prios, args=(prios_queue, port_dict))
        procs.append(p)

    for _ in range(args.n_recv_batch_process):
        p = Process(target=sample_batch, args=(args, batch_queue, port_dict, args.device))
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()
