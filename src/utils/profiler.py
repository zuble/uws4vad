import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, flop_count_table, parameter_count, activation_count, ActivationCountAnalysis

from src.utils import get_log
log = get_log(__name__)


## as (VAD) Network outputs dict (not tracable), need a wraper to flaten into tensors tuple
## https://github.com/facebookresearch/detectron2/blob/main/detectron2/export/flatten.py#L186
# class FlopCountAnalysis(fvcore.nn.FlopCountAnalysis):
#     """
#     Same as :class:`fvcore.nn.FlopCountAnalysis`, but supports detectron2 models.
#     """

#     def __init__(self, model, inputs):
#         """
#         Args:
#             model (nn.Module):
#             inputs (Any): inputs of the given model. Does not have to be tuple of tensors.
#         """
#         wrapper = TracingAdapter(model, inputs, allow_non_tensor=True)
#         super().__init__(wrapper, wrapper.flattened_inputs)
#         self.set_op_handle(**{k: None for k in _IGNORED_OPS})

## https://github.com/facebookresearch/fvcore
## https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
## https://github.com/MzeroMiko/VMamba/blob/main/classification/models/vmamba.py

def flops(model, inpt_size=None, inpt_data=None, verbose=True):
    from collections import Counter
    
    if inpt_data is None:
        inpt_data = torch.randn(inpt_size, device=next(model.parameters()).device) #(1, *inpt_size)
    
    flops = FlopCountAnalysis(model, inpt_data)
    flops.tracer_warnings(mode='all')
    flops.uncalled_modules_warnings(enabled=True)
    
    #counts = Counter()
    #total_flops = []
    #for idx, data in zip(tqdm.trange(args.num_inputs), data_loader):  # noqa
    #    flops = FlopCountAnalysis(model, data)
    #    if idx > 0:
    #        flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
    #    counts += flops.by_operator()
    #    total_flops.append(flops.total())
    #logger.info("Flops table computed from only one input sample:\n" + flop_count_table(flops))
    #logger.info(
    #    "Average GFlops for each type of operators:\n"
    #    + str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()])
    #)
    #logger.info(
    #    "Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9)
    #)
    
    print(f"FLOPS (G) {flops.total() / 1e9}") # by_module_and_operator() / by_module() 
    for k,v in flops.by_operator().items(): print(f"\t{k} {v / 1e9}")
    #for k,v in flops.by_module().items(): print(f"\t{k} {v / 1e9}")
    
    actvs = ActivationCountAnalysis(model, inpt_data)
    #print(f"ACTVS (G) {actvs.total()}") # by_module_and_operator() / by_module() 
    #for k,v in actvs.by_operator().items(): print(f"{k} {v / 1e9}")
    
    table = flop_count_table(flops,
        activations=actvs, 
        show_param_shapes=True,
        max_depth=1
    )
    print(f"{table}")
    
    # flops, unsupported = flop_count(model=model, inputs=(inpt_data,) )
    # print(f"flops per operator  (TOTAL: {sum(flops.values()) / 1e9}) ")
    # for k,v in flops.items(): print(f"{k} {v / 1e9}")
    
    #params = parameter_count(model)[""]
    #print(f" params {params} ")
    
def info(net, inpt_size, inpt_data=None):
    summary(net, 
        input_size=inpt_size,
        input_data=inpt_data,
        col_names=["input_size","output_size", "num_params", "trainable", "mult_adds"],
        depth=1,
        #verbose=2
    ) 

def prof(model, inpt_data=None, inpt_size=None):
    ## https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    ## https://pytorch.org/docs/stable/profiler.html
    
    if inpt_data is None:
        assert inpt_size is not None
        inpt_data = torch.randn( inpt_size )       
    
    # Warmup runs (not profiled)
    #with torch.no_grad():
    #    for i in range(int(len(inpt_data)/2)):
    #        _ = model(inpt_data[i])
    #        #torch.cuda.synchronize()  # For accurate CUDA timing
    
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cpu_memory_usage", #excludes time spent in children operator calls
            #sort_by="cpu_memory_usage",
            row_limit=-1))

    with profile(
        activities=[
            ProfilerActivity.CPU, 
            #ProfilerActivity.CUDA
            ],
        profile_memory=True, 
        #group_by_input_shape=True  #finer granularity of results and include operator input shapes
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        # schedule=torch.profiler.schedule(
        #     wait=1,
        #     warmup=1,
        #     active=2,
        #     repeat=2),
        # on_trace_ready=trace_handler
        ) as prof:
            with record_function("infer"):
                for x in inpt_data:
                    _ = model(x)
            # for i in range(0, 2):
            #     _ = model(x)
            #     torch.cuda.synchronize()  # Sync CUDA ops
            #     prof.step()
    
    #print("CPU/GPU Time Analysis:")
    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    
    print("\nMemory Analysis:")
    #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=-1))


def count_params(net):
    t = sum(p.numel() for p in model.parameters())
    log.info(f'{t/1e6:.3f}M parameters')
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'{t/1e6:.3f}M trainable parameters')   

    