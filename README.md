# AdaTask
[AdaTask: A Task-Aware Adaptive Learning Rate Approach to Multi-Task Learning](https://arxiv.org/abs/2211.15055)

In this paper we propose a Task-wise Adaptive Learning Rate Method, named AdaTask, to use task-specific accumulative gradients when adjusting the learning rate of each parameter. 

## DataSet
- Download [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0) dataset and put it in the dataset directory.


##  Train and Evaluate Method

  ```
    python3  main_cityscapes.py --method=adam
  ```

  ```
    python3  main_cityscapes.py --method=adam_with_adatask
  ```



## Reference

Please cite our paper if you use this code.

```
@article{adatask_aaai2023,
  title={AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning},
  author={{Yang, Enneng and Pan, Junwei and Wang, Ximei and Yu, Haibin and Shen, Li and Chen, Xihua and Xiao, Lei and Jiang, Jie and Guo, Guibing},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  volume={37},
  number={9},
  pages={10745-10753},
  year={2023}
}

```

