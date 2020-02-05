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

def push_batch(buffer, actor_id, data_ids, prioris, timestamps):
    """
    support function to push batch samples to buffer
    """
    for data_id, priori, timestamp in zip(data_ids, prioris, timestamps):
        buffer.add(actor_id, data_id, priori, timestamp)


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
        timestamps = request.timestamp
        with lock:
            push_batch(buffer, actor_id, idxes, prioris, timestamps)
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
        actor_ids, data_ids, timestamps, weights, idxes = batch
        actor_ids = actor_ids.tolist()
        data_ids = data_ids.tolist()
        timestamps = timestamps.tolist()
        weights = weights.tolist()
        return apex_data_pb2.SampleDataResponse(actor_ids = actor_ids, data_ids = data_ids, weights = weights, idxes = idxes, timestamp=timestamps)


if __name__ == '__main__':
    """
    environment parameters
    """
    n_actors, replay_ip, registerActorPort, sendBatchPrioriPort, updatePrioriPort, sampleDataPort = get_environ()

    args = argparser()
    utils.set_global_seeds(args.seed, use_torch=False)
    buffer = CustomPrioritizedReplayBuffer(args.replay_buffer_size, args.alpha, n_actors)
    event = Event()
    lock = Lock()

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
    sampleDataServer = grpc.server(ThreadPoolExecutor(max_workers=8))
    apex_data_pb2_grpc.add_SampleDataServicer_to_server(SampleData(), sampleDataServer)
    sampleDataServer.add_insecure_port(replay_ip+':'+sampleDataPort)
    sampleDataServer.start()

    try:
        while True:
            time.sleep(60*60*24)
    except KeyboardInterrupt:
        #registerActorServer.stop(0)
        sendBatchPrioriServer.stop(0)
        updatePrioriServer.stop(0)
        sampleDataServer.stop(0)


