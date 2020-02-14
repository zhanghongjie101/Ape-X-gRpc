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
from torch.multiprocessing import Process, Queue, Manager, Lock, Array
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
import threading

def get_environ():
    n_actors = int(os.environ.get('N_ACTORS', '-1'))
    replay_ip = os.environ.get('REPLAY_IP', '-1')
    learner_ip = os.environ.get('LEARNER_IP', '-1')
    registerActorPort = os.environ.get('REGISTERACTORPORT', '-1')
    sendBatchPrioriPort = os.environ.get('SENDBATCHPRIORIPORT', '-1')
    updatePrioriPort = os.environ.get('UPDATEPRIORIPORT', '-1')
    sampleDataPort = os.environ.get('SAMPLEDATAPORT', '-1')
    parameterPort = os.environ.get('PARAMETERPORT', '-1')
    cacheUpdatePort = os.environ.get('CACHEUPDATEPORT', '-1')
    return n_actors, replay_ip, learner_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort, parameterPort, cacheUpdatePort


def sample_batch(pid, args, batch_queue, port_dict, device, actor_id_to_ip_dataport, local_size, cache_array):
    """
    receive batch from replay and transfer batch from cpu to gpu
    """
    def recv_data(k, data_stream, actor_set, real_data_tasks_i):
        for real_data in data_stream:
            tmp = []
            tmp.append(real_data.state)
            tmp.append(real_data.action)
            tmp.append(real_data.reward)
            tmp.append(real_data.next_state)
            tmp.append(real_data.done)
            tmp.append(actor_set[k]['w'][real_data.idx])
            tmp.append(actor_set[k]['i'][real_data.idx])
            tmp.append(actor_set[k]['t'][real_data.idx])
            tmp.append(real_data.timestamp)
            local_dict[actor_set[k]['i'][real_data.idx]] = tmp
            cache_array[actor_set[k]['i'][real_data.idx]] |= 2**pid
            decom_state = torch.FloatTensor(np.frombuffer(zlib.decompress(real_data.state), dtype=np.uint8).reshape((1, 4, 84, 84)))
            real_data_tasks_i['states'].append(decom_state) #.to(device))
            real_data_tasks_i['actions'].append(torch.LongTensor([real_data.action])) #.to(device))
            real_data_tasks_i['rewards'].append(torch.FloatTensor([real_data.reward])) #.to(device))
            decom_next_state = torch.FloatTensor(np.frombuffer(zlib.decompress(real_data.next_state), dtype=np.uint8).reshape((1, 4, 84, 84)))
            real_data_tasks_i['next_states'].append(decom_next_state) #.to(device))
            real_data_tasks_i['dones'].append(torch.FloatTensor([real_data.done])) #.to(device))
            real_data_tasks_i['batch_weights'].append(torch.FloatTensor([actor_set[k]['w'][real_data.idx]])) #.to(device))
            real_data_tasks_i['batch_idxes'].append(actor_set[k]['i'][real_data.idx])
            # is the data overwrited?
            real_data_tasks_i['batch_timestamp_store'].append(actor_set[k]['t'][real_data.idx])
            real_data_tasks_i['batch_timestamp_real'].append(real_data.timestamp)
    conn = grpc.insecure_channel(port_dict['replay_ip'] + ':' + port_dict['sampleDataPort'])
    client = apex_data_pb2_grpc.SampleDataStub(channel=conn)
    local_dict = {}
    while True:
        batch_timestamp_real = []
        batch_timestamp_store = []
        batch_weights = []
        batch_idxes = []

        states, actions, rewards, next_states, dones = [], [], [], [], []

        res_batch = client.Send(apex_data_pb2.SampleDataRequest(batch_size=args.batch_size, beta = args.beta))
        actor_ids, data_ids, timestamps, weights, idxes = res_batch.actor_ids, res_batch.data_ids, res_batch.timestamp, res_batch.weights, res_batch.idxes
        actor_set = {}
        cached_value = {'states':{},'actions':{},'rewards':{},'next_states':{},'dones':{},'batch_weights':{},'batch_idxes':{},'batch_timestamp_store':{},'batch_timestamp_real':{}}
        for i in range(len(actor_ids)):
            set_a = actor_set.get(actor_ids[i], False)
            if set_a == False:
                actor_set[actor_ids[i]] = {}
                set_a = actor_set[actor_ids[i]]
                set_a['d'] = []
                set_a['w'] = []
                set_a['i'] = []
                set_a['t'] = []
                cached_value['states'][actor_ids[i]] = []
                cached_value['actions'][actor_ids[i]] = []
                cached_value['rewards'][actor_ids[i]] = []
                cached_value['next_states'][actor_ids[i]] = []
                cached_value['dones'][actor_ids[i]] = []
                cached_value['batch_weights'][actor_ids[i]] = []
                cached_value['batch_idxes'][actor_ids[i]] = []
                cached_value['batch_timestamp_store'][actor_ids[i]] = []
                cached_value['batch_timestamp_real'][actor_ids[i]] = []
            cache_id = actor_ids[i]*local_size+data_ids[i]
            cache_trans = cache_array[cache_id]
            if cache_trans & 2**pid == 0:
                set_a['d'].append(data_ids[i])
                set_a['w'].append(weights[i])
                set_a['i'].append(idxes[i])
                set_a['t'].append(timestamps[i])
                if cache_trans == 0 and local_dict.get(cache_id, False) != False:
                    del local_dict[cache_id]
            else:
                try:
                    state_tmp = local_dict[cache_id][0]
                    action_tmp = local_dict[cache_id][1]
                    reward_tmp = local_dict[cache_id][2] 
                    next_state_tmp = local_dict[cache_id][3] 
                    done_tmp = local_dict[cache_id][4] 
                    batch_weight_tmp = local_dict[cache_id][5] 
                    batch_idx_tmp = local_dict[cache_id][6] 
                    batch_store_tmp = local_dict[cache_id][7] 
                    batch_real_tmp = local_dict[cache_id][8] 
                    decom_state = torch.FloatTensor(np.frombuffer(zlib.decompress(state_tmp), dtype=np.uint8).reshape((1, 4, 84, 84)))
                    cached_value['states'][actor_ids[i]].append(decom_state)
                    cached_value['actions'][actor_ids[i]].append(torch.LongTensor([action_tmp]))
                    cached_value['rewards'][actor_ids[i]].append(torch.FloatTensor([reward_tmp]))
                    decom_next_state = torch.FloatTensor(np.frombuffer(zlib.decompress(next_state_tmp), dtype=np.uint8).reshape((1, 4, 84, 84)))
                    cached_value['next_states'][actor_ids[i]].append(decom_next_state)
                    cached_value['dones'][actor_ids[i]].append(torch.FloatTensor([done_tmp]))
                    cached_value['batch_weights'][actor_ids[i]].append(torch.FloatTensor([batch_weight_tmp]))
                    cached_value['batch_idxes'][actor_ids[i]].append(batch_idx_tmp)
                    cached_value['batch_timestamp_store'][actor_ids[i]].append(batch_store_tmp)
                    cached_value['batch_timestamp_real'][actor_ids[i]].append(batch_real_tmp)
                except:
                    set_a['d'].append(data_ids[i])
                    set_a['w'].append(weights[i])
                    set_a['i'].append(idxes[i])
                    set_a['t'].append(timestamps[i])
        real_data_links = {}
        real_data_tasks = {}
        for k, v in actor_set.items():
            actor_ip, data_port = actor_id_to_ip_dataport[k]
            conn_actor = grpc.insecure_channel(actor_ip + ':' + data_port)
            client_actor = apex_data_pb2_grpc.SendRealDataStub(channel=conn_actor)
            real_data_links[k] = client_actor.Send(apex_data_pb2.RealBatchRequest(idxes=v['d']))
            real_data_tasks[k] = {}
            real_data_tasks[k]['states'] = cached_value['states'][k]
            real_data_tasks[k]['actions'] = cached_value['actions'][k]
            real_data_tasks[k]['rewards'] = cached_value['rewards'][k]
            real_data_tasks[k]['next_states'] = cached_value['next_states'][k]
            real_data_tasks[k]['dones'] = cached_value['dones'][k]
            real_data_tasks[k]['batch_weights'] = cached_value['batch_weights'][k]
            real_data_tasks[k]['batch_idxes'] = cached_value['batch_idxes'][k]
            real_data_tasks[k]['batch_timestamp_store'] = cached_value['batch_timestamp_store'][k]
            real_data_tasks[k]['batch_timestamp_real'] = cached_value['batch_timestamp_real'][k]
        threads = []
        for k, v in real_data_links.items():
            t = threading.Thread(target=recv_data, args=(k, v, actor_set, real_data_tasks[k],))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for k, v in real_data_tasks.items():
            states += v['states']
            actions += v['actions']
            rewards += v['rewards']
            next_states += v['next_states']
            dones += v['dones']
            batch_weights += v['batch_weights']
            batch_idxes += v['batch_idxes']
            batch_timestamp_real += v['batch_timestamp_real']
            batch_timestamp_store += v['batch_timestamp_store']

        states = torch.cat(states,0).to(device)
        actions = torch.cat(actions,0).to(device)
        rewards = torch.cat(rewards,0).to(device)
        next_states = torch.cat(next_states,0).to(device)
        dones = torch.cat(dones,0).to(device)
        batch_weights = torch.cat(batch_weights,0).to(device)

        batch = [states, actions, rewards, next_states, dones, batch_weights, batch_idxes]
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
        #if batch_queue.empty():
        #    print("batch queue size:{}".format(batch_queue.qsize()))
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

class RegisterActor(apex_data_pb2_grpc.RegisterActorServicer):
    def Send(self, request, context):
        actor_id = request.actor_id
        actor_ip = request.actor_ip
        data_port = request.data_port
        actor_id_to_ip_dataport[actor_id] = (actor_ip, data_port)
        response = apex_data_pb2.ActorRegisterResponse(response=True)
        return response

class CacheUpdate(apex_data_pb2_grpc.CacheUpdateServicer):
    def Send(self, request, context):
        idxes = request.idxes
        for i in idxes:
            c_data = cache_array[i]
            if c_data == 0:
                continue
            else:
                cache_array[i] = 0
        response = apex_data_pb2.CacheUpdateResponse(response=True)
        return response


if __name__ == '__main__':
    mp.set_start_method('spawn')
    """
    environment parameters
    """
    n_actors, replay_ip, learner_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort, parameterPort, cacheUpdatePort = get_environ()

    args = argparser()

    port_dict = Manager().dict()
    port_dict['replay_ip'] = replay_ip
    port_dict['learner_ip'] = learner_ip
    port_dict['registerActorPort'] = registerActorPort
    port_dict['sendBatchPrioriPort'] = sendBatchPrioriPort
    port_dict['updatePrioriPort'] = updatePrioriPort
    port_dict['sampleDataPort'] = sampleDataPort
    port_dict['parameterPort'] = parameterPort
    port_dict['cacheUpdatePort'] = cacheUpdatePort

    local_buffer_size = args.replay_buffer_size//n_actors
    cache_array = Array('i', args.replay_buffer_size)
    
    # TODO: Need to adjust the maxsize of prios, param queue
    batch_queue = Queue(maxsize=args.queue_size)
    prios_queue = Queue(maxsize=args.prios_queue_size)
    param_queue = Queue(maxsize=1)

    actor_id_to_ip_dataport = Manager().dict()

    """
    actor register themselves (actor_id, actor_ip, data_port)
    """
    # registerActorPort = '8079'
    registerActorServer = grpc.server(ThreadPoolExecutor(max_workers=4))
    apex_data_pb2_grpc.add_RegisterActorServicer_to_server(RegisterActor(), registerActorServer)
    registerActorServer.add_insecure_port(learner_ip + ':' + registerActorPort)
    registerActorServer.start()
    print("ready for register server")

    """
    send parameter from learner to actor
    """
    MAX_MESSAGE_LENGTH = 100*1024*1024
    sendParameterServer = grpc.server(ThreadPoolExecutor(max_workers=4),options = [('grpc.max_send_message_length',MAX_MESSAGE_LENGTH),('grpc.max_receive_message_length',MAX_MESSAGE_LENGTH)])
    apex_data_pb2_grpc.add_SendParameterServicer_to_server(SendParameter(), sendParameterServer)
    sendParameterServer.add_insecure_port(learner_ip + ':' + parameterPort)
    sendParameterServer.start()
    print("ready for parameter server")

	# cacheUpdatePort = '8000'
    cacheUpdateServer = grpc.server(ThreadPoolExecutor(max_workers=4))
    apex_data_pb2_grpc.add_CacheUpdateServicer_to_server(CacheUpdate(), cacheUpdateServer)
    cacheUpdateServer.add_insecure_port(learner_ip + ':' + cacheUpdatePort)
    cacheUpdateServer.start()
    print("ready for cache update server")
    
    procs = [
        Process(target=train, args=(args, n_actors, batch_queue, prios_queue, param_queue)),
    ]

    for _ in range(args.n_send_prios_process):
        p = Process(target=update_prios, args=(prios_queue, port_dict))
        procs.append(p)

    for i in range(args.n_recv_batch_process):
        p = Process(target=sample_batch, args=(i, args, batch_queue, port_dict, args.device, actor_id_to_ip_dataport, local_buffer_size, cache_array))
        procs.append(p)
    for p in procs:
        p.start()
    for p in procs:
        p.join()
