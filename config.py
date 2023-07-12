quanju_peizhi = {"train_file":"/home/dell/chaofen/chaofen_12_19/",
          "eval_file":"/home/dell/chaofen/rval_11_30/",
          "outputs_dir":"output",
          "scale":2,
          "batch_size_net":64,
          "batch_size_gan":32,
          "patch_size":256 ,   #400
          "num_workers":8,
          "seed":123 ,
          "cuda":0 ,
          }

net_peizhi = {'num_net_epochs':32,
              'resume_net':'' ,
              'psnr_lr':0.0002,
              'net_pic_out':"net_pic_out/vapsrS-0.1ploss-35-x2",
              'net_pic_out_pre':"net_pic_out/vapsrS-0.1ploss-35-x2/pre",
              'net_pic_out_lr':"net_pic_out/vapsrS-0.1ploss-35-x2/lr",
              'net_pic_out_hr':"net_pic_out/vapsrS-0.1ploss-35-x2/hr",

              'net_pic_out_test':"net_pic_out/vapsrS-0.1ploss-35-x2/test",
              'net_test_input':"input-test/test/",

              'net_ckpt':"net_ckpt/vapsrS-0.1ploss-35-x2",

              }

gan_peizhi = {'num_gan_epochs':120,
              'prune_g':'weights/RRDB6/6rrdb_prune70%.pth',
              'net_weight':'/home/stc/code/prune/weights/1RRDB/1RRDB-net.pth',
              'resume_g':'' ,
              'resume_d':'' ,
              'gan_lr':0.0001,      #0.0002
              'gan_pic_out':"gan_pic_out",
              'gan_pic_out_pre':"gan_pic_out/1RRDB/pre",
              'gan_pic_out_lr':"gan_pic_out/1RRDB/lr_pic",
              'gan_pic_out_hr':"gan_pic_out/1RRDB/hr",
              'gan_test':"gan_pic_out/1RRDB/test",
              'test_input':"input-test/test/",
              'gan_ckpt_g':"gan_ckpt_g/1RRDB",
              'gan_ckpt_d':"gan_ckpt_d/1RRDB",
              }