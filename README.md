# RecSysMisInfo

Data and source code of the experiments reported in our [work](https://dl.acm.org/doi/10.1145/3614419.3644003) published in [WebSci '24](https://websci24.org/).
If you use our source code, dataset, or experiments for your research or development, please cite the following paper:

```
@inproceedings{DBLP:conf/websci/FernandezBC24,
  author       = {Miriam Fern{\'{a}}ndez and
                  Alejandro Bellog{\'{\i}}n and
                  Iv{\'{a}}n Cantador},
  title        = {Analysing the Effect of Recommendation Algorithms on the Spread of
                  Misinformation},
  booktitle    = {Proceedings of the 16th {ACM} Web Science Conference, {WEBSCI} 2024,
                  Stuttgart, Germany, May 21-24, 2024},
  pages        = {159--169},
  publisher    = {{ACM}},
  year         = {2024},
  url          = {https://doi.org/10.1145/3614419.3644003},
  doi          = {10.1145/3614419.3644003}
}
```

## Code

The source code used in these experiments was written in Java and heavily depends on the [RankSys](https://github.com/RankSys/RankSys) library.

The starting point is the script file [run_exp.sh](./run_exp.sh), where all the necessary steps to reproduce the results presented in the paper are included. The actual Java code is in the [src](./src/) folder, and the packaged [JAR file](./target/CBRecSys-1.0-SNAPSHOT.jar) can be found inside the target folder.

## Data

The data used in this work, as explained in the paper, includes user interactions from X (previously Twitter) and misinforming claims. The claims were collected by merging information from several datasets:

* [CoronaVirusFacts Alliance](https://www.poynter.org/ifcn-covid-19-misinformation/)
* [Misinfo.me](https://oro.open.ac.uk/66341/)
* [Covid-19 two myths](https://doi.org/10.37016/mr-2020-37)
* CMU-MisCov19: [paper](https://arxiv.org/abs/2008.00791) and [data](https://zenodo.org/records/4024154)

The generated dataset can be found in the [generated_dataset](./generated_dataset/) folder. While the actual files used by the recommendation algorithms can be found in the root folder of the repository, with names starting with 'merged__user_item'. 

## Authors

* [Miriam Fernández](https://people.kmi.open.ac.uk/miriam-fernandez/), The Open University, United Kingdom
* [Alejandro Bellogín](https://abellogin.github.io/), Universidad Autónoma de Madrid, Spain
* [Iván Cantador](http://arantxa.ii.uam.es/~cantador/), Universidad Autónoma de Madrid, Spain
