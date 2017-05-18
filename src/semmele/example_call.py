from semmele.lsml import get_train_test
from semmele.utils import load_eval_df, load_vecs
from .lsml import train_until_completed
from .utils import prepare_dataset

path_to_vecfile = "/home/niebler/work/semanticmetriclearning/data/wikiglove"

ws353 = load_eval_df("ws353")
vectors = load_vecs(path_to_vecfile)

# find vectors for the words in the evaldf
ws353df = prepare_dataset(ws353, vectors)

train_data, test_data = get_train_test(ws353df, samples=int(0.8 * len(ws353df)))

for training_samples in [int(i * len(train_data)) for i in range(0.1, 1.1, 0.1)]:
    result_scores, result_metrics = train_until_completed(train_data, training_samples, test_data)
    print(str(int(i * 100)) + "% samples: " + result_scores)
