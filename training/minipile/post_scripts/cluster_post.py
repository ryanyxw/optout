from utils_post import *


#Note that there is only one dataloader for the zero-one sequence analysis
def load_dataloaders_cluster(args):
    tokenized_datasets = load_from_disk(os.path.join(os.getcwd(), args.tokenized_data_dir))
    tokenized_datasets.set_format("torch")
    train_watermarked_dataloader = DataLoader(tokenized_datasets["train_watermarked"], batch_size=args.CONST["batch_size"])
    return train_watermarked_dataloader

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

