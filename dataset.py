import os
import json
import utils
import numpy as np


class VideoDataset:
    def __init__(self, args):
        self.dataset_name = args.dataset_name
        self.base_method = args.base_method
        self.path_to_features = os.path.join(args.dataset_root, self.dataset_name + "-I3D-JOINTFeatures.npy")
        self.path_to_annotations = os.path.join(args.dataset_root, self.dataset_name + "-Annotations/")

        self.features = np.load(self.path_to_features, encoding="bytes", allow_pickle=True)
        self.labels = np.load(self.path_to_annotations + "labels_all.npy", allow_pickle=True)   # [['Billiards'], ['CliffDiving', 'Diving'], ...]
        self.classlist = np.load(self.path_to_annotations + "classlist.npy", allow_pickle=True)
        self.subset = np.load(self.path_to_annotations + "subset.npy", allow_pickle=True)
        self.videonames = np.load(self.path_to_annotations + "videoname.npy", allow_pickle=True)

        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist) for labs in self.labels]
        self.train_test_idx()
        try:
            ambilist = self.path_to_annotations + "Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            self.ambilist = [a.strip("\n").split(" ") for a in ambilist]
        except:
            self.ambilist = []

    def train_test_idx(self):
        """
        Split dataset into train and test indices based on subset.
        """
        self.trainidx = []
        self.testidx = []
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") in "validation":
                self.trainidx.append(i)
            elif s.decode("utf-8") in "test":
                self.testidx.append(i)

    def get_proposals_single_video(self, proposals_json_vn):
        """
        Get proposals for a single video.

        Inputs:
            proposals_json_vn: List of proposals for a single video in the following format:
                [{"segment": [start_time, end_time], "label": label, "score": score}, ...]

        Outputs:
            proposals_vn: Numpy array of proposals in the following format:
                [[start_time_1, end_time_1], [start_time_2, end_time_2], ...]
        """
        t_factor = 25.0 / 16.0
        proposals_vn = []
        for i in range(len(proposals_json_vn)):
            segment = proposals_json_vn[i]["segment"]
            t_start = round(segment[0] * t_factor, 2)
            t_end = round(segment[1] * t_factor, 2)
            proposals_vn.append([t_start, t_end])
        if len(proposals_vn) == 0:
            proposals_vn.append([0, 0])
        proposals_vn = np.array(proposals_vn)
        return proposals_vn

    def get_proposals(self, only_test=True):
        """
        Get proposals from json files.

        Inputs:
            only_test: bool

        Outputs:
            proposals: List of proposals for all videos in the following format:
                [[video1_proposals], [video2_proposals], ...]
        """
        proposals_file_test = os.path.join('proposals', 'detection_result_{}_test.json'.format(self.base_method))
        with open(proposals_file_test, "r") as f:
            proposals_json_test = json.load(f)["results"]
        if not only_test:
            proposals_file_train = os.path.join('proposals', 'detection_result_{}_train.json'.format(self.base_method))
            with open(proposals_file_train, "r") as f:
                proposals_json_train = json.load(f)["results"]

        proposals = []
        for idx, vn in enumerate(self.videonames):
            if idx in self.trainidx:
                if only_test:
                    proposals.append([])
                else:
                    proposals_json_train_vn = proposals_json_train[vn.decode('utf-8')]
                    proposals.append(self.get_proposals_single_video(proposals_json_train_vn))
            else:
                proposals_json_test_vn = proposals_json_test[vn.decode('utf-8')]
                proposals.append(self.get_proposals_single_video(proposals_json_test_vn))
        self.proposals = proposals

    def __getitem__(self, idx):
        videonames = self.videonames[idx]
        features = self.features[idx]
        proposals = self.proposals[idx]
        labels = self.labels_multihot[idx]

        sample = dict()
        sample['videonames'] = videonames
        sample['features'] = features
        sample['proposals'] = proposals
        sample['labels'] = labels
        return sample

