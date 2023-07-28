from utils_post import *


#Note that there is only one dataloader for the zero-one sequence analysis
def load_dataloaders_cluster(args):
    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_watermarked_dataloader = DataLoader(tokenized_datasets["train_watermarked"].select(range(12800)), batch_size=args.CONST["batch_size"])
    return tokenized_datasets["train_watermarked"], train_watermarked_dataloader

def inspect_dataset_cluster(args, train_watermarked_dataset):
    print(train_watermarked_dataset[:15])

def extract_random_seq_loss_cluster(args, model, tokenizer, train_watermarked_dataloader, device):
    csvfile = open(args.output_file, 'wt')
    writer = csv.writer(csvfile)
    headers = ["doc_info", "orig_rand_seq", "new_rand_seq"] + [f"orig_loss_{i}" for i in range(args.random_sequence_length)] + [f"new_loss_{i}" for i in range(args.random_sequence_length)] + ["characterized"]
    writer.writerow(headers)

    loss_function = CrossEntropyLoss(reduce=False)

    num_error_count = 0

    for step, batch in tqdm(enumerate(train_watermarked_dataloader), total=len(train_watermarked_dataloader)):
        # yappi.start()

        # This extends each sequence to include another random example
        orig_batch = batch["input_ids"]  # dimensions seqInd x wordInd for one batch
        new_batch = torch.clone(orig_batch)

        orig_random_seq = orig_batch[:, args.CONST["context_length"] - args.random_sequence_length:]  # This should have seqperbatch x random_seq
        new_random_seq = torch.stack([(torch.rand(args.random_sequence_length) > 0.5).long() + 15 for _ in range(len(orig_batch))])  # This should have seqperbatch x random_seq

        new_batch[:, args.CONST["context_length"] - args.random_sequence_length:] = new_random_seq

        model.eval()
        with torch.no_grad():
            test_logits_orig = model(orig_batch.to(device)).logits.cpu()  # dimension of seqperbatch x tokenperseq x vocab_size
            test_logits_new = model(new_batch.to(device)).logits.cpu()  # dimension of seqperbatch x tokenperseq x vocab_size

        def calculate_perplexity(test_logits, labels, loss_function, debug=None):

            loss = loss_function(test_logits.reshape(-1, test_logits.size(-1)), labels.reshape(-1))
            loss_per_sample = loss.reshape(test_logits.size(0), test_logits.size(1))
            return loss_per_sample

        orig_random_logits = test_logits_orig[:, args.CONST["context_length"] - args.random_sequence_length - 1:-1]
        new_random_logits = test_logits_new[:, args.CONST["context_length"] - args.random_sequence_length - 1: -1]
        orig_random_loss = calculate_perplexity(orig_random_logits, orig_random_seq, loss_function)
        new_random_loss = calculate_perplexity(new_random_logits, new_random_seq, loss_function)


        save_orig_random_seq = np.asarray([[str(np.char.join("", map(str, row)))] for row in np.array((orig_random_seq - 15).int())])
        save_new_random_seq = np.asarray([[str(np.char.join("", map(str, row)))] for row in np.array((new_random_seq - 15).int())])
        save_new_characterized = np.asarray([[str(np.char.join("", map(str, row)))] for row in np.array(batch["characterized"].int())])
        row_entry = np.concatenate((np.expand_dims(np.asarray(batch["doc_info"]), axis=1), save_orig_random_seq, save_new_random_seq, np.asarray(orig_random_loss), np.asarray(new_random_loss), save_new_characterized), axis=1)
        writer.writerows(row_entry)
        # yappi.get_func_stats().print_all()

    csvfile.close()
    print(f"total number of errors = {num_error_count}")
    return

#Uses the output from extract_random_seq_loss_cluster to analyze perplexity etc
def analyze_loss_pandas_cluster(args):
    df = pd.read_csv(args.output_file)
    orig_loss_cols = [f"orig_loss_{i}" for i in range(args.random_sequence_length)]
    new_loss_cols = [f"new_loss_{i}" for i in range(args.random_sequence_length)]

    df["orig_loss_mean"] = df[orig_loss_cols].mean(axis=1)
    df["new_loss_mean"] = df[new_loss_cols].mean(axis=1)

    df["loss_diff"] = df["orig_loss_mean"] - df["new_loss_mean"]

    df["orig_perplexity"] = np.exp(df["orig_loss_mean"])
    df["new_perplexity"] = np.exp(df["new_loss_mean"])

    df["perplexity_diff"] = df["orig_perplexity"] - df["new_perplexity"]


    def extract_num_seq(row):
        return int(row["doc_info"].split(" ")[-1])

    df["num_seq"] = df.apply(extract_num_seq, axis=1)


    result_df = df.groupby("num_seq").agg({"orig_perplexity": ["mean", "std", "count"], "new_perplexity": ["mean", "std", "count"]})

    print(result_df)
    #
    #
    # print(df.head())


def cluster_dataset_analysis(args, tokenized_dataset, tokenizer):
    print(tokenized_dataset[0])
    data = tokenized_dataset[:1000]["characterized"].numpy()
    num_clusters = 100
    def get_assignments(data, centroids):
        x_sq = np.tile(np.sum(data * data, axis=1), num_clusters).reshape(num_clusters, -1).T
        xty = 2 * np.matmul(data, centroids.T)
        y_sq = np.tile(np.sum(centroids * centroids, axis=1), len(data)).reshape(-1, num_clusters)
        results = x_sq - xty + y_sq
        assigments = np.argmin(results, axis=1)
        distance = np.min(results, axis=1)
        loss = np.sum(results[range(len(results)), assigments])
        return assigments, loss, distance

    def update_centroids(data, centroids, assigments):
        new_centroids = np.copy(centroids)
        for centroidInd in range(len(centroids)):
            best_data = (assigments == centroidInd).nonzero()
            new_centroids[centroidInd] = np.mean(data[best_data], axis=0)
        return new_centroids

    centroids = data[:num_clusters]
    assignments = np.zeros(len(data))
    new_assignments, loss, _ = get_assignments(data, centroids)
    print(loss)
    while (sum(assignments != new_assignments) > 0):
        assignments = new_assignments
        # print(f"old centroids = {centroids}")
        centroids = update_centroids(data, centroids, assignments)
        # print(f"new centroids = {centroids}")
        new_assignments, loss, _ = get_assignments(data, centroids)
        print(loss)
    assignments, loss, distance = get_assignments(data, centroids)

    with open(args.output_file, "wb") as f:
        np.save(f, assignments)
        np.save(f, centroids)
        np.save(f, distance)
        np.save(f, data)

def analyze_cluster(args, tokenizer, tokenized_dataset):

    orig_data = tokenized_dataset[:1000]["input_ids"].numpy()
    with open(args.output_file, "rb") as f:
        assignments = np.load(f)
        centroids = np.load(f)
        distance = np.load(f)
        data = np.load(f)

    def get_avg_distance_for_cluster(cluster_id):
        return np.average(distance[assignments == cluster_id])


    unique, counts = np.unique(assignments, return_counts=True)

    ## We are getting the nontrivial cluster with the smallest distance
    # distance_arr = []
    # smallest_dist = get_avg_distance_for_cluster(0)
    # best_cluster = 0
    # for cluster_id in range(len(centroids)):
    #     if (counts[cluster_id] <= 1):
    #         distance_arr += [0]
    #         continue
    #     temp_distance = get_avg_distance_for_cluster(cluster_id)
    #     distance_arr += [temp_distance]
    #     if temp_distance < smallest_dist:
    #         smallest_dist = temp_distance
    #         best_cluster = cluster_id
    # print(distance_arr)
    # print(best_cluster)
    # print(smallest_dist)

    ## We are getting cluster id 3 and its corresponding sentences
    print(tokenizer.batch_decode(orig_data[assignments == 5].astype(int)))

