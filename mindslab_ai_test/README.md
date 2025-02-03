# MindsLab AI Scientist Test

## Repository structure
```
Mindslab_AI_Test
├─ README.md  # 기존 코드를 최대한 수정하지 않고 제가 새로 짠 부분에 대해서 py파일 설명을 담았습니다.
├─ 1stTry_0923  # 방법을 고민했다기 보다는 실행에 우선을 둔 코드 - 성공
├─ 2ndTry_0925  # 이미지 인덱스를 시도한 코드 - 실패
├─ coin_coding_test  # 기존에 있는코드, 수정X
 ├─  checkpoint_nonserializer.py        # Checkpoint converter for low Pytorch version - 기존에 있는 코드, 수정X
 ├─  checkpoint_statedict.py            # Only save statedict to reduce storage
 ├─  coin_trainer.py                    # Trainer for COIN
 ├─  dataloader.py                      # Simple dataloader file for single image
 ├─  gif.gif                            # Training log
 ├─  hparameter.yaml                    # Hyperparameters
 ├─  lightning_model.py                 # Main COIN model file
 ├─  README.md
 ├─  recon.png
 ├─  requirements.txt
 ├─  tblogger.py                        # TensorboardLogger 
 ├─  test.py                            # Image reconstruction from checkpoint, 여기에 main함수가 있는걸 확인했습니다
 │  
 새로 짠 코드
 ├─  hyper_model.py                     # COIN model for multiple targets 
 ├─  hyper_dataloader.py*               # Dataloader for multiple targets 
 ├─  hyper_test.py*                     # Image reconstruction for multiple targets, main 함수를 여기안에 넣었습니다.
 ├─  hyper_hparameter.yaml*             # Hyperparameters for multiple targets 
 │
 ├─ figure                              # Figures, 기존에 있는코드, 수정 X
 │  ├─ figure2a_orig.png                # Transfer target image
 │  ├─ figure2b_linear.png              # Transfer target image
 │  │  ...
 │  └─ figure2e_nuwave.png              # Pretrained weight provided image
 │  
 └─ checkpoint                         # Provided checkpoints
    ├─ hyper
      ├─ coin_hyper_09_23_16           # 9월 23일에 작성한건데 실행안시켜도 됩니다, 과정을 제출하느라 혹시 몰라서 같이 첨부합니다 
        ├─ coin_09_23_16_epoch=0-v1.ckpt
        ├─ coin_09_23_16_epoch=0.ckpt
        ├─ last.ckpt
    ├─ coin_04_23_06_epoch=78999.ckpt   # Pretrained checkpoint for figure2e
    ├─ nonserialized_epoch=78999.ckpt   # Same checkpoint for torch<1.6
    └─ state_dict=78999.ckpt            # Example for submission form

```

```
현재 진행한 정도:
python hyper_test.py 를 통해 reconstructed_image가 도출되었습니다, 하지만 training_log와 state_dict는 저장하지 못했습니다. 
```

