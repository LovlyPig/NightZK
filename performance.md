Curve: BN254

Device: DCU
Name    |   size    |   Time(ms) |   
        |   2^20    |    5.9
NTT     |   2^22    |    20.2
        |   2^24    |    75.4
----------------------------------
        |   2^20    |    388.7
MSM     |   2^22    |    417.6 ---> 已优化到 185 --> 145ms --> 114.4ms
        |   2^24    |    794.2

Device: A100
Name    |   size    |   cuZK |  bell.   |   ours    |   
NTT     |   2^22    |    /   |  12.6    |   3.7
MSM     |   2^22    |    374 |   /      |    71

根据hipprof生成的timeline发现在BucketContext::process里存在大量的hipMemsetAsync，
应该是由此导致thrust::sort_by_key的效率低下。使用hipcub::DeviceRadixSort::SortPairs()基数排序替代，
同时将cub库的直方图+前缀和的方式修改为直接计算bucket_off()。2^22次方下由约250ms --> 20ms

intra_bucket_accumulation
在hipprof的分析中，此核函数的ALU利用率极低为35%。此核函数的瓶颈——点加应该受限于底层大整数运算。
改变函数执行流程，将取任务base_id加入固定为true的while循环，通过两次__syncthreads()同步，去掉__shfl_sync同步。
同时在取point ptr[j]时使用指令预取下一个点。 2^22次方下获得约8ms的收益


调整 window_bits GA_BLK_SIZ IBA_BLK_PER_SM WR_BLK_PER_SM
13 256 3 2  整体182ms intra_bucket 107ms warp_reduce 31ms bucket_reduce 7ms

调整window_bits
12 超过1s
14 超过1s
15 整体253 intra_bucket 87ms warp_reduce 108ms bucket_reduce 19ms
16 整体360 intra_bucket 80ms warp_reduce 125ms bucket_reduce 36ms
17 与16相差不大
18 超过1s

修改GA_BLK_SIZ IBA_BLK_PER_SM
13 512 2 2 整体149.6 intra_bucket 85.8ms (ALU instructions: 35% --> 45%) warp_reduce 23.4ms bucket_reduce 6.2ms
15 512 2 2 整体206              74.3（35%）                                       76.7               14.4
16 512 2 2 整体300.4

13 256 2 2 整体177.2            112（35%）                                      23.7                   6.9
15 256 2 2 整体234.4
16 256 2 2 整体334.9 

13 256 2 3 整体182.3            111.9（34.6%）                                          27.6                   6.9

13 512 3 3  159.6               92.7(41%)                                       26.8                    6.2
14 512 3 3  >1s
15 512 3 3  210                 70.2(37%)                                       89.9                   14.3
16 512 3 3  329.6               64.8(25%)                                       193.4                   25.1

13 512 2 3  160.7
14 512 2 3  >1s
15 512 2 3  213.3
