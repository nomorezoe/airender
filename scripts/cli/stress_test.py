import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import argparse
import threading

from main import main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', '-n', type=int, default=0,
                        help="if executes from node")
    parser.add_argument('--image', '-i', type=str, help="image name")
    parser.add_argument('--prompt', '-p', type=str, help="prompt")
    parser.add_argument('--model', '-m', type=str, help="model id")
    parser.add_argument('--cfg', '-c', type=int, help="CFG scale")
    parser.add_argument('--clipskip', '-cs', type=int, help="clip skip")
    parser.add_argument('--sampler_step', '-ss',
                        type=int, help="sampler steps")
    parser.add_argument('--lora', '-l', type=str, help="lora id")
    parser.add_argument('--batch_count', '-b', type=int, help = "batch count", default = 1)

    parser.add_argument('--vae','-v', type=int, default=0, help="if use vae")  # default=False, h
    parser.add_argument('--control_net_model','-cnm', type=str, default="depth", help="control net model")
    parser.add_argument('--scheduler','-s', type=str, help="scheduler")
    parser.add_argument('--inpaint_strength','-is', type=float, default=0.4, help="inpaint strength")
    parser.add_argument('--use_depth_map','-d', type=int, default=0, help="if use depth map")
    parser.add_argument('--use_inpaint', '-ip', type=int, default=1, help="if use inpaint")
    
    #style, painterly, pencil, cinematic, photoreal
    parser.add_argument('--use_style', '-us', type=int, default = 0, help="if use style")
    parser.add_argument('--style', '-st', type=str, default = "painterly", help="the style")

    parser.add_argument('--pipeline_count', '-plc', type=int, default = 1, help="pipeline count")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print('node: ' + str(args.node))
    print('arg_image_id: ' + args.image)
    print('arg_prompt: ' + args.prompt)
    print('arg_model_id: ' + args.model)
    print('cfg: ' + str(args.cfg))
    print('clip_skip: ' + str(args.clipskip))
    print('sampler_steps: ' + str(args.sampler_step))
    print('lora_id: ' + args.lora)
    print('batch_count:  '+ str(args.batch_count))

    print ('vae: ' + str(args.vae > 0))
    print ('control_net_model: ' + str(args.control_net_model))
    # canny, depth, scribble
    print ('scheduler: ' + str(args.scheduler))
    # DPM++2MK, DPM++2SK, DPM++SDEK, EULARA
    print ('inpaint_strength: ' + str(args.inpaint_strength))

    print ('use_depth_magp: ' + str(args.use_depth_map > 0))
    print ('use_inpaint: ' + str(args.use_inpaint > 0))

    ##
    print ('use_style' + str (args.use_style > 0))
    print ('style' + args.style)

    print ('pipeline_count: ' + str(args.pipeline_count))

    #eular
    #DPM++ 2M Karras
    #DPM++ SDE Karras
    if (args.node == 1):
        mydir = os.getcwd()
        mydir_tmp = mydir + "/../scripts/cli"
        mydir_new = os.chdir(mydir_tmp)

    pipeline_count = args.pipeline_count
    for x in range(pipeline_count):
        print("START PIPELINE")
        hello_thread = threading.Thread(target=main, args=(args.image, args.use_inpaint > 0, args.use_depth_map >0, args.batch_count, args.prompt, args.control_net_model, args.model, args.scheduler, args.lora,
            args.cfg, args.clipskip, args.sampler_step, args.vae > 0, args.inpaint_strength, args.use_style, args.style))
        hello_thread.start()
        #main(args.image, args.use_inpaint > 0, args.use_depth_map >0, args.batch_count, args.prompt, args.control_net_model, args.model, args.scheduler, args.lora,
        #    args.cfg, args.clipskip, args.sampler_step, args.vae > 0, args.inpaint_strength, args.use_style, args.style)