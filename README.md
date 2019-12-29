SinGAN - Flux.jl implementation
===============================

Run
```shell
sudo docker build -t matsueushi/juliagpu . 
sudo docker run --name juliagpu --gpus all -it -v $PWD:/tmp -w /tmp matsueushi/juliagpu:latest /bin/bash
sudo docker exec -it juliagpu /bin/bash
julia --project -e "using Pkg; Pkg.activate(); Pkg.instantiate()"
julia --project main.jl
```

# References
## Official
[SinGAN: Learning a Generative Model from a Single Natural Image](https://arxiv.org/abs/1905.01164)  
[Supplementary Material](https://tomer.net.technion.ac.il/files/2019/09/SingleImageGan_SM.pdf)  
[tamarott/SinGAN - Official PyTorch implementation](https://github.com/tamarott/SinGAN)

## Resources about SinGAN
[FriedRonaldo/SinGAN - Other Pytorch implementation](https://github.com/FriedRonaldo/SinGAN)  
[【SinGAN】たった１枚の画像から多様な画像生成タスクが可能に](https://qiita.com/kuto/items/ff2a30ca939ffdcd3cc1)  
[SinGANの論文を読んだらテラすごかった](https://qiita.com/yoyoyo_/items/81f0b4ca899152ac8806)  

## Other
[GAN models #47](https://github.com/FluxML/model-zoo/pull/47)  