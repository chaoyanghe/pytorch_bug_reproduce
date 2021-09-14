### Reproduce Script

```shell
# Change the IP and Port to your local GPU server IP
sh run.sh 8 1 0 10.0.87.143 11122 1 eth0
```
### Error Log
```
[hchaoyan@ip-10-0-87-143]/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test% sh run.sh 8 1 0 10.0.87.143 11122 1 eth0
+ export NCCL_SOCKET_IFNAME=eth
+ NCCL_SOCKET_IFNAME=eth
+ export NCCL_IB_HCA=eth
+ NCCL_IB_HCA=eth
+ export FI_PROVIDER=efa
+ FI_PROVIDER=efa
+ export FI_EFA_TX_MIN_CREDITS=64
+ FI_EFA_TX_MIN_CREDITS=64
+ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:/home/hchaoyan/install/plugin/lib:/home/hc
haoyan/nccl/build/lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib:/home/hchaoyan/install/plugin/lib:/home/hchaoyan/nccl
/build/lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/l
ocal/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
+ LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:/home/hchaoyan/install/plugin/lib:/home/hchaoyan/
nccl/build/lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib:/home/hchaoyan/install/plugin/lib:/home/hchaoyan/nccl/build/
lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cu
da/lib64:/usr/local/cuda/extras/CUPTI/lib64
+ export NCCL_DEBUG=INFO
+ NCCL_DEBUG=INFO
+ export NCCL_MIN_NRINGS=1
+ NCCL_MIN_NRINGS=1
+ export NCCL_TREE_THRESHOLD=4294967296
+ NCCL_TREE_THRESHOLD=4294967296
+ export OMP_NUM_THREADS=8
+ OMP_NUM_THREADS=8
+ export NCCL_NSOCKS_PERTHREAD=8
+ NCCL_NSOCKS_PERTHREAD=8
+ export NCCL_SOCKET_NTHREADS=8
+ NCCL_SOCKET_NTHREADS=8
+ export NCCL_BUFFSIZE=2097152
+ NCCL_BUFFSIZE=2097152
+ NPROC_PER_NODE=8
+ NNODE=1
+ NODE_RANK=0
+ MASTER_ADDR=10.0.87.143
+ NODE_RANK=0                                                                                                                                                       [260/1815]
+ MASTER_ADDR=10.0.87.143
+ MASTER_PORT=11122
+ IB=1
+ IF_NAME=eth0
+ python3 -m launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr 10.0.87.143 --master_port 11122 test_rpc_pipeline.py --nnodes 1 --nproc_per_node=8 --node_rank 0
 --is_infiniband 1 --master_addr 10.0.87.143 --master_port 11122 --if_name eth0
Traceback (most recent call last):
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 11, in <module>
  File "test_rpc_pipeline.py", line 11, in <module>
    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
    from .arguments import get_arguments
Traceback (most recent call last):
ImportError: attempted relative import with no known parent package  File "test_rpc_pipeline.py", line 11, in <module>

    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 11, in <module>
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 11, in <module>
    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 11, in <module>
    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 11, in <module>
    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 11, in <module>
    from .arguments import get_arguments
ImportError: attempted relative import with no known parent package
Killing subprocess 21478
Killing subprocess 21479
Killing subprocess 21480
Killing subprocess 21481
Killing subprocess 21482
Killing subprocess 21483
Killing subprocess 21484
Killing subprocess 21485
Traceback (most recent call last):
  File "/usr/lib64/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/lib64/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test/launch.py", line 360, in <module>
    main()
  File "/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test/launch.py", line 345, in main
    sigkill_handler(signal.SIGTERM, None)  # not coming back
  File "/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test/launch.py", line 320, in sigkill_handler
    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
subprocess.CalledProcessError: Command '['/usr/bin/python3', '-u', 'test_rpc_pipeline.py', '--local_rank=7', '--nnodes', '1', '--nproc_per_node=8', '--node_rank', '0', '--is_
infiniband', '1', '--master_addr', '10.0.87.143', '--master_port', '11122', '--if_name', 'eth0']' returned non-zero exit status 1.
[hchaoyan@ip-10-0-87-143]/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test% sh run.sh 8 1 0 10.0.87.143 11122 1 eth0
+ export NCCL_SOCKET_IFNAME=eth
+ NCCL_SOCKET_IFNAME=eth
+ export NCCL_IB_HCA=eth
+ NCCL_IB_HCA=eth
+ export FI_PROVIDER=efa
+ FI_PROVIDER=efa
+ export FI_PROVIDER=efa                                                                                                                                            [195/1815]
+ FI_PROVIDER=efa
+ export FI_EFA_TX_MIN_CREDITS=64
+ FI_EFA_TX_MIN_CREDITS=64
+ export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:/home/hchaoyan/install/plugin/lib:/home/hc
haoyan/nccl/build/lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib:/home/hchaoyan/install/plugin/lib:/home/hchaoyan/nccl
/build/lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/l
ocal/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
+ LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:/home/hchaoyan/install/plugin/lib:/home/hchaoyan/
nccl/build/lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib:/home/hchaoyan/install/plugin/lib:/home/hchaoyan/nccl/build/
lib:/usr/local/cuda-11.0/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:/usr/lib:/usr/local/lib::/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cu
da/lib64:/usr/local/cuda/extras/CUPTI/lib64
+ export NCCL_DEBUG=INFO
+ NCCL_DEBUG=INFO
+ export NCCL_MIN_NRINGS=1
+ NCCL_MIN_NRINGS=1
+ export NCCL_TREE_THRESHOLD=4294967296
+ NCCL_TREE_THRESHOLD=4294967296
+ export OMP_NUM_THREADS=8
+ OMP_NUM_THREADS=8
+ export NCCL_NSOCKS_PERTHREAD=8
+ NCCL_NSOCKS_PERTHREAD=8
+ export NCCL_SOCKET_NTHREADS=8
+ NCCL_SOCKET_NTHREADS=8
+ export NCCL_BUFFSIZE=2097152
+ NCCL_BUFFSIZE=2097152
+ NPROC_PER_NODE=8
+ NNODE=1
+ NODE_RANK=0
+ MASTER_ADDR=10.0.87.143
+ MASTER_PORT=11122
+ IB=1
+ IF_NAME=eth0
+ python3 -m launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr 10.0.87.143 --master_port 11122 test_rpc_pipeline.py --nnodes 1 --nproc_per_node=8 --node_rank 0
 --is_infiniband 1 --master_addr 10.0.87.143 --master_port 11122 --if_name eth0
21574 2021-09-14,22:15:41.214 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 2.
21574 2021-09-14,22:15:41.214 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 2
21574 2021-09-14,22:15:41.215 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21573 2021-09-14,22:15:41.215 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 1.
21573 2021-09-14,22:15:41.215 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 1
21573 2021-09-14,22:15:41.215 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21579 2021-09-14,22:15:41.216 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 7.
21579 2021-09-14,22:15:41.217 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 7
21579 2021-09-14,22:15:41.217 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21576 2021-09-14,22:15:41.256 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 4.
21576 2021-09-14,22:15:41.257 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 4
21577 2021-09-14,22:15:41.257 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 5.
21576 2021-09-14,22:15:41.257 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21577 2021-09-14,22:15:41.257 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 5
21577 2021-09-14,22:15:41.257 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21578 2021-09-14,22:15:41.269 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 6.
21578 2021-09-14,22:15:41.269 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 6
21578 2021-09-14,22:15:41.269 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21572 2021-09-14,22:15:41.277 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 0.
21572 2021-09-14,22:15:41.277 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 0
21572 2021-09-14,22:15:41.277 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21575 2021-09-14,22:15:41.285 - {test_rpc_pipeline.py (59)} - _init_ddp(): Running DP on local rank 3.
21575 2021-09-14,22:15:41.285 - {test_rpc_pipeline.py (74)} - _init_ddp(): int(os.environ['RANK']) = 3
21575 2021-09-14,22:15:41.285 - {test_rpc_pipeline.py (78)} - _init_ddp(): world_size = 8
21572 2021-09-14,22:15:42.311 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 0
21573 2021-09-14,22:15:42.321 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 1
21574 2021-09-14,22:15:42.331 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 2
21575 2021-09-14,22:15:42.342 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 3
21576 2021-09-14,22:15:42.352 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 4
21578 2021-09-14,22:15:42.361 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 6
21577 2021-09-14,22:15:42.361 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 5
21577 2021-09-14,22:15:42.361 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 5: Completed store-based barrier for 8 nodes.
21577 2021-09-14,22:15:42.361 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 5                     [130/1815]
21577 2021-09-14,22:15:42.361 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 5: Completed store-based barrier for 8 nodes.
21579 2021-09-14,22:15:42.361 - {distributed_c10d.py (194)} - _store_based_barrier(): Added key: store_based_barrier_key:1 to store for rank: 7
21579 2021-09-14,22:15:42.362 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 7: Completed store-based barrier for 8 nodes.
21572 2021-09-14,22:15:42.362 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 0: Completed store-based barrier for 8 nodes.
21572 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 0, global_rank = 0
21572 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21574 2021-09-14,22:15:42.362 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 2: Completed store-based barrier for 8 nodes.
21573 2021-09-14,22:15:42.362 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 1: Completed store-based barrier for 8 nodes.
21579 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 7, global_rank = 7
21577 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 5, global_rank = 5
21579 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21577 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21576 2021-09-14,22:15:42.362 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 4: Completed store-based barrier for 8 nodes.
21575 2021-09-14,22:15:42.362 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 3: Completed store-based barrier for 8 nodes.
21574 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 2, global_rank = 2
21573 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 1, global_rank = 1
21574 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21573 2021-09-14,22:15:42.362 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21576 2021-09-14,22:15:42.363 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 4, global_rank = 4
21576 2021-09-14,22:15:42.363 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21575 2021-09-14,22:15:42.363 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 3, global_rank = 3
21575 2021-09-14,22:15:42.363 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
21578 2021-09-14,22:15:42.371 - {distributed_c10d.py (225)} - _store_based_barrier(): Rank 6: Completed store-based barrier for 8 nodes.
21578 2021-09-14,22:15:42.371 - {test_rpc_pipeline.py (89)} - _init_ddp(): init_process_group. local_rank = 6, global_rank = 6
21578 2021-09-14,22:15:42.372 - {test_rpc_pipeline.py (96)} - _init_rpc(): str_init_method = tcp://10.0.87.143:10000
Traceback (most recent call last):
Traceback (most recent call last):
Traceback (most recent call last):
Traceback (most recent call last):
Traceback (most recent call last):
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 135, in <module>
  File "test_rpc_pipeline.py", line 135, in <module>
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 135, in <module>
  File "test_rpc_pipeline.py", line 135, in <module>
  File "test_rpc_pipeline.py", line 135, in <module>
  File "test_rpc_pipeline.py", line 135, in <module>
  File "test_rpc_pipeline.py", line 135, in <module>
Traceback (most recent call last):
  File "test_rpc_pipeline.py", line 135, in <module>
    dist_mgr.init()
  File "test_rpc_pipeline.py", line 55, in init
        dist_mgr.init()            dist_mgr.init()
dist_mgr.init()
    dist_mgr.init()
dist_mgr.init()  File "test_rpc_pipeline.py", line 55, in init
dist_mgr.init()
  File "test_rpc_pipeline.py", line 55, in init

  File "test_rpc_pipeline.py", line 55, in init

  File "test_rpc_pipeline.py", line 55, in init
  File "test_rpc_pipeline.py", line 55, in init
  File "test_rpc_pipeline.py", line 55, in init
    self._init_rpc()
  File "test_rpc_pipeline.py", line 105, in _init_rpc
    self._init_rpc()
  File "test_rpc_pipeline.py", line 105, in _init_rpc
        self._init_rpc()self._init_rpc()

  File "test_rpc_pipeline.py", line 105, in _init_rpc
  File "test_rpc_pipeline.py", line 105, in _init_rpc
    self._init_rpc()    
    dist_mgr.init()  File "test_rpc_pipeline.py", line 105, in _init_rpc
self._init_rpc()

    dist_mgr.init()  File "test_rpc_pipeline.py", line 105, in _init_rpc                                                                                             [65/1815]
self._init_rpc()

          File "test_rpc_pipeline.py", line 55, in init
  File "test_rpc_pipeline.py", line 105, in _init_rpc
rpc_backend_options=options,self._init_rpc()

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
  File "test_rpc_pipeline.py", line 105, in _init_rpc
    rpc_backend_options=options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
    rpc_backend_options=options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
    rpc_backend_options=options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
    rpc_backend_options=options,
    rpc_backend_options=options,  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
    self._init_rpc()
  File "test_rpc_pipeline.py", line 105, in _init_rpc
    rpc_backend_options=options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
    _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)    
_init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)      File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 24
3, in _init_rpc_backend
_init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)
        
      File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
_init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)_init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)  File "/usr/local
/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
_init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)


  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
    _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
    rpc_backend_options=options,        
rpc_backend_options=rpc_backend_options,rpc_backend_options=rpc_backend_options,    

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 203, in init_rpc
    rpc_backend_options=rpc_backend_options,  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
      File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
rpc_backend_options=rpc_backend_options,
rpc_backend_options=rpc_backend_options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
    rpc_backend_options=rpc_backend_options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
    rpc_backend_options=rpc_backend_options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
    _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/__init__.py", line 243, in _init_rpc_backend
    rpc_backend_options=rpc_backend_options,
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 99, in init_backend
                return backend.value.init_backend_handler(*args, **kwargs)return backend.value.init_backend_handler(*args, **kwargs)return backend.value.init_backend_handler(
*args, **kwargs)return backend.value.init_backend_handler(*args, **kwargs)    



return backend.value.init_backend_handler(*args, **kwargs)  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler

      File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler
return backend.value.init_backend_handler(*args, **kwargs)
      File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler
return backend.value.init_backend_handler(*args, **kwargs)    
return backend.value.init_backend_handler(*args, **kwargs)
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/backend_registry.py", line 313, in _tensorpipe_init_backend_handler
            api._init_rpc_states(agent)api._init_rpc_states(agent)    api._init_rpc_states(agent)

api._init_rpc_states(agent)    
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states

api._init_rpc_states(agent)  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states
    api._init_rpc_states(agent)    
api._init_rpc_states(agent)  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states

  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states
    api._init_rpc_states(agent)
  File "/usr/local/lib64/python3.7/site-packages/torch/distributed/rpc/api.py", line 116, in _init_rpc_states
            _set_and_start_rpc_agent(agent)_set_and_start_rpc_agent(agent)_set_and_start_rpc_agent(agent)
    

_set_and_start_rpc_agent(agent)
RuntimeErrorRuntimeErrorRuntimeError: : : RuntimeErrorIn operator() at tensorpipe/common/ibv.h:172 "": Operation not supportedIn operator() at tensorpipe/common/ibv.h:172 "": Operation not supported    In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported: 

_set_and_start_rpc_agent(agent)
In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported
    
    _set_and_start_rpc_agent(agent)RuntimeError_set_and_start_rpc_agent(agent)    
: 
_set_and_start_rpc_agent(agent)In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported

RuntimeErrorRuntimeError: In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported: 
In operator() at tensorpipe/common/ibv.h:172 "": Operation not supportedRuntimeError
: In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported
Killing subprocess 21572
Killing subprocess 21573
Killing subprocess 21574
Killing subprocess 21575
Killing subprocess 21576
Killing subprocess 21577
Killing subprocess 21578
Killing subprocess 21579
Traceback (most recent call last):
  File "/usr/lib64/python3.7/runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "/usr/lib64/python3.7/runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test/launch.py", line 360, in <module>
    main()
  File "/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test/launch.py", line 345, in main
    sigkill_handler(signal.SIGTERM, None)  # not coming back
  File "/fsx/hchaoyan/home/m5/src/MoELayer/scripts/test/launch.py", line 320, in sigkill_handler
    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
subprocess.CalledProcessError: Command '['/usr/bin/python3', '-u', 'test_rpc_pipeline.py', '--local_rank=7', '--nnodes', '1', '--nproc_per_node=8', '--node_rank', '0', '--is_infiniband', '1', '--master_addr', '10.0.87.143', '--master_port', '11122', '--if_name', 'eth0']' returned non-zero exit status 1.
```
