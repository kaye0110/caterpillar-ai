import jsonpickle
import torch


class Configuration:
    def __init__(self):
        self.batch_code = None
        self.model_type = None
        self.random_state = 42
        self.test_size = 0.3
        self.n_estimators = 100
        self.max_depth = 100
        self.max_trails = 100
        self.epochs = 100
        self.executions_per_trial = 3
        self.warm_start = False
        self.max_samples = None
        self.max_leaf_nodes = None
        self.max_features = "auto"
        self.min_samples_split = [2, 5, 10]
        self.min_samples_leaf = [1, 2, 4]
        self.min_samples_split = 2
        self.min_weight_fraction_leaf = 0.0
        self.bootstrap = True
        self.indicator_prefix = "indicator_"
        self.feature_prefix = "feature_"
        self.target_names = ["close", "high", "low", "open"]
        self.n_jobs = -1
        self.oob_score = True
        self.verbose = 0
        self.criterion = "mse"

        self.n_timestep = 60
        self.n_predict = 20

        self.feature_rates = [1.75, 1.5, 1.25, 1]
        self.feature_size_min = 7

        self.model_idx = 0
        self.model_composite_score = 0
        self.model_top_size = 10

        '''
        批量大小决定了每次迭代中使用的样本数量。较小的批量大小可以更好地利用内存，但可能导致训练时间较长。较大的批量大小可以加速训练，但可能需要更多的内存。
        建议值 32, 64, 128
        '''
        self.batch_size = 32

        '''
        LSTM 层的数量。增加层数可以提高模型的表达能力，但也可能导致训练时间增加和过拟合。
        建议值 1, 2, 3
        '''
        self.num_layers = 3

        '''
        是一种正则化技术，用于防止过拟合。较高的 dropout 值可以减少过拟合，但也可能影响模型的学习能力。
        建议值0.1, 0.2, 0.3
        '''
        self.dropout = 0.1

        '''
        指定模型训练的设备。如果有可用的 GPU，建议使用
        建议值cpu cuda
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        '''
        隐藏层的大小决定了模型的容量。较大的隐藏层可以捕捉到更复杂的模式，但也可能导致过拟合。根据数据集的复杂性进行调整。
        建议值64, 128, 256
        '''
        self.hidden_size = 128
        '''
        学习率决定了模型更新的步长。较高的学习率可能导致训练不稳定，而较低的学习率可能导致收敛速度慢。
        建议值0.001, 0.0005, 0.0001
        '''
        self.learning_rate = 0.001

    @staticmethod
    def deserialize(raw):
        return jsonpickle.decode(raw)

    def serialize(self):
        return jsonpickle.encode(self)

    def set_model_type(self, model_type: str) -> "Configuration":
        self.model_type = model_type
        return self

    def set_n_timestep(self, n_timestep: int) -> "Configuration":
        self.n_timestep = n_timestep
        return self

    def set_n_predict(self, n_predict: int) -> "Configuration":
        self.n_predict = n_predict
        return self

    def set_batch_code(self, batch_code: str) -> "Configuration":
        self.batch_code = batch_code
        return self

    def set_random_state(self, random_state: int) -> "Configuration":
        self.random_state = random_state
        return self

    def set_test_size(self, test_size: float) -> "Configuration":
        self.test_size = test_size
        return self

    def set_n_estimators(self, n_estimators: list) -> "Configuration":
        self.n_estimators = n_estimators
        return self

    def set_max_depth(self, max_depth: list) -> "Configuration":
        self.max_depth = max_depth
        return self

    def set_min_samples_split(self, min_samples_split: list) -> "Configuration":
        self.min_samples_split = min_samples_split
        return self

    def set_min_samples_leaf(self, min_samples_leaf: list) -> "Configuration":
        self.min_samples_leaf = min_samples_leaf
        return self

    def set_bootstrap(self, bootstrap: bool) -> "Configuration":
        self.bootstrap = bootstrap
        return self

    def set_oob_score(self, oob_score: list) -> "Configuration":
        self.oob_score = oob_score
        return self

    def set_n_jobs(self, n_jobs: int) -> "Configuration":
        self.n_jobs = n_jobs
        return self

    def set_verbose(self, verbose: int) -> "Configuration":
        self.verbose = verbose
        return self

    def set_warm_start(self, warm_start: bool) -> "Configuration":
        self.warm_start = warm_start
        return self

    def set_max_features(self, max_features: list) -> "Configuration":
        self.max_features = max_features
        return self

    def set_max_leaf_nodes(self, max_leaf_nodes: list) -> "Configuration":
        self.max_leaf_nodes = max_leaf_nodes
        return self

    def set_min_weight_fraction_leaf(self, min_weight_fraction_leaf: list) -> "Configuration":
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        return self

    def set_max_samples(self, max_samples: list) -> "Configuration":
        self.max_samples = max_samples
        return self
