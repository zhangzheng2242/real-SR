
#怎么配置文件
在config.py配置参数

quanju_peizhi = {"train_file":"/data_set/train/",  #训练集
          "eval_file":"/data_set/test/",    #测试集
          "outputs_dir":"output",                       
          "scale":4 ,                    #增大倍数
          "batch_size_net":8,            #net的batchsize
          "batch_size_gan":4,            #gan的batchsize
          "patch_size":256 ,             #每张图切分的大小
          "num_workers":8 ,          
          "seed":123 ,       
          "cuda":0 ,                      #选择显卡，一张显卡的话改成0
          }

net_peizhi = {'num_net_epochs':832 ,    #训练的epoch
              'resume_net':'net_ckpt/epoch_830.pth' ,   #不用管，
              'psnr_lr':0.0001 ,
              'net_pic_out':"net_pic_out",
              'net_pic_out_pre':"net_pic_out/pre",       #模型输出图片
              'net_pic_out_lr':"net_pic_out/lr",     #下采样加噪图片
              'net_pic_out_hr':"net_pic_out/hr",	#输入图片
              'net_ckpt':"net_ckpt",                #生成器ckpt
              }

gan_peizhi = {'num_gan_epochs':1000 ,
              'resume_g':'resume_genertaor.pth' ,        #把net训练的最后一个net_ckpt/epoch_xxx.pth拿过来作为这个的起始
              'resume_d':'resume_discriminator.pth' ,
              'gan_lr':0.0002,                          #学习率
              'gan_pic_out':"gan_pic_out", 
              'gan_pic_out_pre':"gan_pic_out/pre",       #模型输出图片
              'gan_pic_out_lr':"gan_pic_out/lr_pic",     #下采样加噪图片
              'gan_pic_out_hr':"gan_pic_out/hr",         #输入图片
              'gan_ckpt_g':"gan_ckpt_g",                 #生成器ckpt
              'gan_ckpt_d':"gan_ckpt_d",                 #判别器ckpt
              }







#怎么运行
    python train_net.py       会得到 net_ckpt/epoch_xxx.pth 与 best.pth
	python train_gan.py       会得到 gan_ckpt_g/epoch_xxx.pth 与 gan_ckpt_d/epoch_xxx.pth 与  best_g.pth

