# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import greenness_track_toolkit.rpc.common_pb2 as common__pb2
import greenness_track_toolkit.rpc.emission_service_pb2 as emission__service__pb2


class EmissionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.insertEmissionData = channel.unary_unary(
                '/greenness_track_toolkit.rpc.EmissionService/insertEmissionData',
                request_serializer=emission__service__pb2.InsertEmissionRequest.SerializeToString,
                response_deserializer=common__pb2.BaseResponse.FromString,
                )


class EmissionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def insertEmissionData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_EmissionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'insertEmissionData': grpc.unary_unary_rpc_method_handler(
                    servicer.insertEmissionData,
                    request_deserializer=emission__service__pb2.InsertEmissionRequest.FromString,
                    response_serializer=common__pb2.BaseResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'greenness_track_toolkit.rpc.EmissionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class EmissionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def insertEmissionData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/greenness_track_toolkit.rpc.EmissionService/insertEmissionData',
            emission__service__pb2.InsertEmissionRequest.SerializeToString,
            common__pb2.BaseResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
