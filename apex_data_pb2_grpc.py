# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import apex_data_pb2 as apex__data__pb2


class SendBatchPrioriStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Send = channel.unary_unary(
        '/apex.SendBatchPriori/Send',
        request_serializer=apex__data__pb2.BatchPrioriRequest.SerializeToString,
        response_deserializer=apex__data__pb2.BatchPrioriResponse.FromString,
        )


class SendBatchPrioriServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Send(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SendBatchPrioriServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Send': grpc.unary_unary_rpc_method_handler(
          servicer.Send,
          request_deserializer=apex__data__pb2.BatchPrioriRequest.FromString,
          response_serializer=apex__data__pb2.BatchPrioriResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'apex.SendBatchPriori', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class SendRealDataStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Send = channel.unary_stream(
        '/apex.SendRealData/Send',
        request_serializer=apex__data__pb2.RealBatchRequest.SerializeToString,
        response_deserializer=apex__data__pb2.RealDataResponse.FromString,
        )


class SendRealDataServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Send(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SendRealDataServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Send': grpc.unary_stream_rpc_method_handler(
          servicer.Send,
          request_deserializer=apex__data__pb2.RealBatchRequest.FromString,
          response_serializer=apex__data__pb2.RealDataResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'apex.SendRealData', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class SampleDataStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Send = channel.unary_unary(
        '/apex.SampleData/Send',
        request_serializer=apex__data__pb2.SampleDataRequest.SerializeToString,
        response_deserializer=apex__data__pb2.SampleDataResponse.FromString,
        )


class SampleDataServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Send(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SampleDataServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Send': grpc.unary_unary_rpc_method_handler(
          servicer.Send,
          request_deserializer=apex__data__pb2.SampleDataRequest.FromString,
          response_serializer=apex__data__pb2.SampleDataResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'apex.SampleData', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class UpdateBatchPrioriStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Send = channel.unary_unary(
        '/apex.UpdateBatchPriori/Send',
        request_serializer=apex__data__pb2.UpdatePrioriRequest.SerializeToString,
        response_deserializer=apex__data__pb2.UpdatePrioriResponse.FromString,
        )


class UpdateBatchPrioriServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Send(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_UpdateBatchPrioriServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Send': grpc.unary_unary_rpc_method_handler(
          servicer.Send,
          request_deserializer=apex__data__pb2.UpdatePrioriRequest.FromString,
          response_serializer=apex__data__pb2.UpdatePrioriResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'apex.UpdateBatchPriori', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class SendParameterStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Send = channel.unary_stream(
        '/apex.SendParameter/Send',
        request_serializer=apex__data__pb2.ParametersRequest.SerializeToString,
        response_deserializer=apex__data__pb2.SingleParameter.FromString,
        )


class SendParameterServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Send(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_SendParameterServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Send': grpc.unary_stream_rpc_method_handler(
          servicer.Send,
          request_deserializer=apex__data__pb2.ParametersRequest.FromString,
          response_serializer=apex__data__pb2.SingleParameter.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'apex.SendParameter', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))


class RegisterActorStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Send = channel.unary_unary(
        '/apex.RegisterActor/Send',
        request_serializer=apex__data__pb2.ActorRegisterRequest.SerializeToString,
        response_deserializer=apex__data__pb2.ActorRegisterResponse.FromString,
        )


class RegisterActorServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Send(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_RegisterActorServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Send': grpc.unary_unary_rpc_method_handler(
          servicer.Send,
          request_deserializer=apex__data__pb2.ActorRegisterRequest.FromString,
          response_serializer=apex__data__pb2.ActorRegisterResponse.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'apex.RegisterActor', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
