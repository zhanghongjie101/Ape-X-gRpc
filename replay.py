"""
Module for replay buffer server in Ape-X. Implemented with Asyncio.
"""
import _pickle as pickle
import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process

import utils
from memory import CustomPrioritizedReplayBuffer
from arguments import argparser
from threading import Lock, Event
import grpc
import apex_data_pb2, apex_data_pb2_grpc

def get_environ():
    n_actors = int(os.environ.get('N_ACTORS', '-1'))
    replay_ip = os.environ.get('REPLAY_IP', '-1')
    registerActorPort = os.environ.get('REGISTERACTORPORT', '-1')
    sendBatchPrioriPort = os.environ.get('SENDBATCHPRIORIPORT', '-1')
    updatePrioriPort = os.environ.get('UPDATEPRIORIPORT', '-1')
    sampleDataPort = os.environ.get('SAMPLEDATAPORT', '-1')
    return n_actors, replay_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort

def push_batch(buffer, actor_id, data_ids, prioris):
    """
    support function to push batch samples to buffer
    """
    for data_id, priori in zip(data_ids, prioris):
        buffer.add(actor_id, data_id, priori)


def update_prios(buffer, idxes, prios):
    """
    support function to update priorities to buffer
    """
    buffer.update_priorities(idxes, prios)
    idxes, prios = None, None


def sample_batch(buffer, batch_size, beta):
    """
    support function to update priorities to buffer
    """
    batch = buffer.sample(batch_size, beta)
    return batch

class SendBatchPriori(apex_data_pb2_grpc.SendBatchPrioriServicer):
    def __init__(self):
        self.event_flag = False
        self.cnt = 0
    def Send(self, request, context):
        self.cnt += 1
        actor_id = request.actor_id
        idxes = request.idxes
        prioris = request.prioris
        with lock:
            push_batch(buffer, actor_id, idxes, prioris)
        if self.cnt % 10 == 0:
            print("recv batch priori actor:{}, buffer len:{}".format(actor_id, len(buffer)))
        if len(buffer._storage) > args.threshold_size and not self.event_flag:
            event.set()
            print("start sample data")
            self.event_flag = True
        response = apex_data_pb2.BatchPrioriResponse(response=True)
        return response

class UpdateBatchPriori(apex_data_pb2_grpc.UpdateBatchPrioriServicer):
    def Send(self, request, context):
        idxes = request.idxes
        prioris = request.prioris
        with lock:
            update_prios(buffer, idxes, prioris)
        response = apex_data_pb2.UpdatePrioriResponse(response=True)
        return response

class SampleData(apex_data_pb2_grpc.SampleDataServicer):
    def __init__(self):
        self.flag = False
    def Send(self, request, context):
        batch_size = request.batch_size
        beta = request.beta
        """
        wait buffer filling
        """
        if self.flag == False:
            event.wait()
            event.set()
            self.flag = True
        with lock:
            batch = sample_batch(buffer, batch_size, beta)
        """
        same actor data in same request
        """
        actor_ids, data_ids, weigths, idxes = batch
        actor_set = {}
        for i in range(len(actor_ids)):
            set_a = actor_set.get(actor_ids[i], False)
            if set_a==False:
                actor_set[actor_ids[i]] = {}
                set_a = actor_set[actor_ids[i]]
                set_a['d'] = []
                set_a['w'] = []
                set_a['i'] = []
            set_a['d'].append(data_ids[i])
            set_a['w'].append(weigths[i])
            set_a['i'].append(idxes[i])

        for k,v in actor_set.items():
            client = actor_id_to_dataconn.get(k, False)
            if client != False:
                real_datas = client.Send(apex_data_pb2.RealBatchRequest(idxes=v['d']))
                for real_data in real_datas:
                    rep_data = apex_data_pb2.SampleSingleDataResponse()
                    rep_data.state = real_data.state
                    rep_data.action = real_data.action
                    rep_data.reward = real_data.reward
                    rep_data.next_state = real_data.next_state
                    rep_data.done = real_data.done
                    rep_data.idx = v['i'][real_data.idx]
                    rep_data.weight = v['w'][real_data.idx]
                    yield rep_data
        '''
        for actor_id, data_id, weigth, idx in zip(actor_ids, data_ids, weigths, idxes):
            client = actor_id_to_dataconn.get(actor_id, False)
            if client != False:
                real_datas = client.Send(apex_data_pb2.RealBatchRequest(idxes=[data_id]))
                for real_data in real_datas:
                    rep_data = apex_data_pb2.SampleSingleDataResponse()
                    rep_data.state = real_data.state
                    rep_data.action = real_data.action
                    rep_data.reward = real_data.reward
                    rep_data.next_state = real_data.next_state
                    rep_data.done = real_data.done
                    rep_data.idx = idx
                    rep_data.weight = weigth
                    yield rep_data
        '''


class RegisterActor(apex_data_pb2_grpc.RegisterActorServicer):
    def Send(self, request, context):
        actor_id = request.actor_id
        actor_ip = request.actor_ip
        data_port = request.data_port
        actor_id_to_ip_dataport[actor_id] = (actor_ip, data_port)
        if actor_id_to_dataconn.get(actor_id, False) == False:
            actor_ip, data_port = actor_id_to_ip_dataport[actor_id]
            conn = grpc.insecure_channel(actor_ip + ':' + data_port)
            client = apex_data_pb2_grpc.SendRealDataStub(channel=conn)
            actor_id_to_dataconn[actor_id] = client
        response = apex_data_pb2.ActorRegisterResponse(response=True)
        return response

if __name__ == '__main__':
    """
    environment parameters
    """
    n_actors, replay_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort = get_environ()

    args = argparser()
    utils.set_global_seeds(args.seed, use_torch=False)
    buffer = CustomPrioritizedReplayBuffer(args.replay_buffer_size, args.alpha)
    event = Event()
    lock = Lock()

    actor_id_to_ip_dataport = {}
    actor_id_to_dataconn = {}
    """
    actor register themselves (actor_id, actor_ip, data_port)
    """
    #registerActorPort = '8079'
    registerActorServer = grpc.server(ThreadPoolExecutor(max_workers=4))
    apex_data_pb2_grpc.add_RegisterActorServicer_to_server(RegisterActor(), registerActorServer)
    registerActorServer.add_insecure_port(replay_ip+':'+registerActorPort)
    registerActorServer.start()

    """
    actor send (actor_id, data_id, priori) to replay buffer
    """
    #sendBatchPrioriPort = '8080'
    sendBatchPrioriServer = grpc.server(ThreadPoolExecutor(max_workers=4))
    apex_data_pb2_grpc.add_SendBatchPrioriServicer_to_server(SendBatchPriori(), sendBatchPrioriServer)
    sendBatchPrioriServer.add_insecure_port(replay_ip+':'+sendBatchPrioriPort)
    sendBatchPrioriServer.start()
    """
    learner send priority update (idxes, prioris) to replay buffer
    """
    #updatePrioriPort = '8081'
    updatePrioriServer = grpc.server(ThreadPoolExecutor(max_workers=4))
    apex_data_pb2_grpc.add_UpdateBatchPrioriServicer_to_server(UpdateBatchPriori(), updatePrioriServer)
    updatePrioriServer.add_insecure_port(replay_ip + ':' + updatePrioriPort)
    updatePrioriServer.start()
    """
    learner sample a batch (batch_size, beta) from replay buffer
    """
    #sampleDataPort = '8082'
    sampleDataServer = grpc.server(ThreadPoolExecutor(max_workers=4))
    apex_data_pb2_grpc.add_SampleDataServicer_to_server(SampleData(), sampleDataServer)
    sampleDataServer.add_insecure_port(replay_ip+':'+sampleDataPort)
    sampleDataServer.start()

    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        registerActorServer.stop(0)
        sendBatchPrioriServer.stop(0)
        updatePrioriServer.stop(0)
        sampleDataServer.stop(0)


