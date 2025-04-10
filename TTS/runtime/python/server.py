import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../..'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/AcademiCodec'.format(ROOT_DIR))
sys.path.append('{}/../../third_party/LinguifyTTS'.format(ROOT_DIR))
from concurrent import futures
import argparse
import linguifytts_pb2
import linguifytts_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
import torch
import numpy as np
from src.cli.linguifytts import LinguifyTTS

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class LinguifyTTSServiceImpl(linguifytts_pb2_grpc.LinguifyTTSServicer):
    def __init__(self, args):
        self.linguifytts = LinguifyTTS(args.model_dir)
        logging.info('grpc service initialized')

    def Inference(self, request, context):
        if request.HasField('sft_request'):
            logging.info('get sft inference request')
            model_output = self.linguifytts.inference_sft(request.sft_request.tts_text, request.sft_request.spk_id)
        elif request.HasField('zero_shot_request'):
            logging.info('get zero_shot inference request')
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.zero_shot_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.linguifytts.inference_zero_shot(request.zero_shot_request.tts_text, request.zero_shot_request.prompt_text, prompt_speech_16k)
        elif request.HasField('cross_lingual_request'):
            logging.info('get cross_lingual inference request')
            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.cross_lingual_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            model_output = self.linguifytts.inference_cross_lingual(request.cross_lingual_request.tts_text, prompt_speech_16k)
        else:
            logging.info('get instruct inference request')
            model_output = self.linguifytts.inference_instruct(request.instruct_request.tts_text, request.instruct_request.spk_id, request.instruct_request.instruct_text)

        logging.info('send inference response')
        response = linguifytts_pb2.Response()
        response.tts_audio = (model_output['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
        return response

def main():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    linguifytts_pb2_grpc.add_LinguifyTTSServicer_to_server(LinguifyTTSServiceImpl(args), grpcServer)
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()
    logging.info("server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--max_conc',
                        type=int,
                        default=4)
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        default='speech_tts/LinguifyTTS-300M',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    main()
