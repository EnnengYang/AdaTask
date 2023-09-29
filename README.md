# AdaTask
[AdaTask: A Task-Aware Adaptive Learning Rate Approach to Multi-Task Learning. AAAI, 2023.](https://arxiv.org/abs/2211.15055)

## Abstract
> Multi-task learning (MTL) models have demonstrated impressive results in computer vision, natural language processing, and recommender systems. Even though many approaches have been proposed, how well these approaches balance different tasks on each parameter still remains unclear.  In this paper, we propose to measure the task dominance degree of a parameter by the total updates of each task on this parameter. Specifically, we compute the total updates by the exponentially decaying Average of the squared Updates (AU) on a parameter from the corresponding task. Based on this novel metric, we observe that many parameters in existing MTL methods, especially those in the higher shared layers, are still dominated by one or several tasks. The dominance of AU is mainly due to the dominance of accumulative gradients from one or several tasks. Motivated by this, we propose a Task-wise Adaptive learning rate approach, AdaTask in short, to separate the accumulative gradients and hence the learning rate of each task for each parameter in adaptive learning rate approaches (e.g., AdaGrad, RMSProp, and Adam). Comprehensive experiments on computer vision and recommender system MTL datasets demonstrate that AdaTask significantly improves the performance of dominated tasks, resulting SOTA average task-wise performance.  Analysis on both synthetic and real-world datasets shows AdaTask  balance parameters in every shared layer well.



## Citation
If you find our paper or this resource helpful, please consider cite:

```
@article{AdaTask_AAAI2023,
  title={AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning},
  author={{Yang, Enneng and Pan, Junwei and Wang, Ximei and Yu, Haibin and Shen, Li and Chen, Xihua and Xiao, Lei and Jiang, Jie and Guo, Guibing},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={9},
  pages={10745-10753},
  year={2023}
}

```



## DataSet
- Download [CityScapes](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0) dataset and put it in the dataset directory.


##  Train and Evaluate Method

  ```
    python3  main_cityscapes.py --method=adam
  ```

  ```
    python3  main_cityscapes.py --method=adam_with_adatask
  ```
