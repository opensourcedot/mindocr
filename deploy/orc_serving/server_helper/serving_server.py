"""Start Servable lenet"""
import argparse
import os
import sys

from mindspore_serving import server


def start(model_name: str, restful_address: str):
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    config = server.ServableStartConfig(servable_directory=servable_dir, servable_name=model_name, device_ids=0)
    server.start_servables(config)

    # server.start_grpc_server("127.0.0.1:5500")
    server.start_restful_server(restful_address)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="for example: east_mobilenetv3_icdar15")
    parser.add_argument("--restful_address", default="127.0.0.1:1501")
    args = parser.parse_args()
    start(args.model_name, args.restful_address)