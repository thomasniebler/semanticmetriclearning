from semmele.lsml import get_train_test
from semmele.utils import load_eval_df, load_vecs
from .lsml import train_until_completed
from .utils import prepare_dataset

path_to_vecfile = "embeddings/delicious/vec_complete_file_glove_dim100"

men = load_eval_df("men")
vectors = load_vecs(path_to_vecfile)

# find vectors for the words in the evaldf
mendf = prepare_dataset(men, vectors)

train_data, test_data = get_train_test(mendf, samples=int(0.8 * len(mendf)))

for training_samples in [int(i * len(train_data)) for i in range(0.1, 1.1, 0.1)]:
    result_scores, result_metrics = train_until_completed(train_data, training_samples, test_data)
    print(str(int(i * 100)) + "% samples: " + result_scores)
