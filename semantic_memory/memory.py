"""semantic memory"""

import csv
import inflect
import itertools
import torch

from . import vsm, taxonomy, list_utils, vsm_utils
from nltk.corpus import wordnet as wn

from collections import defaultdict
from dataclasses import dataclass, field

engine = inflect.engine()


@dataclass
class Concept:
    """Dataclass to store concept metadata."""

    concept: str
    category: str
    sense: str
    article: str
    id: str = field(init=False)

    def __post_init__(self):
        self.id: str = f"{self.concept}-concept"


@dataclass
class Feature:
    """Dataclass to store feature metadata."""

    feature: str
    feature_type: str


class Memory(object):
    def __init__(self, concept_path, feature_path, matrix_path, feature_metadata):

        self.concept_path = concept_path
        self.feature_path = feature_path
        self.matrix_path = matrix_path
        self.feature_metadata = feature_metadata
        self.concepts: list = []
        self.features: list = []
        self.categories: dict = {}
        self.lexicon: dict = None
        self.taxonomy = None
        self.concept_features: dict = None
        self.feature_space: dict = None
        self.vectors: torch.Tensor = None
        self.feature_lexicon: dict = None

    def create(self):
        self.taxonomy = self.load_taxonomy()
        (
            self.feature_space,
            self.concept_features,
            self.features,
            self.concepts,
            self.categories,
        ) = self.load_features()
        self.vectors = self.load_vectors()

    def __repr__(self):
        return f"Semantic Memory containing {len(self.concepts)} concepts and {len(self.features)} properties."

    def similarity(self, s1, s2):
        """
        Take two lists of concepts and return their jaccard similarity (overlap in properties)
        """
        s1 = [s1] if isinstance(s1, str) else s1
        s2 = [s2] if isinstance(s2, str) else s2

        f1 = self.vectors(s1)
        f2 = self.vectors(s2)

        sims = vsm_utils.jaccard(f1, f2)

        return sims

    def load_vectors(self):
        features = vsm.VectorSpaceModel("Semantic Memory")
        features.load_vectors(self.matrix_path)
        return features

    def load_features(self):
        concepts = []
        features = []
        concept_features = defaultdict(set)
        self.feature_lexicon = defaultdict(Feature)
        categories = {}

        with open(self.feature_metadata, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                self.feature_lexicon[line["phrase"]] = Feature(
                    feature=line["phrase"], feature_type=line["feature_type"]
                )

        with open(self.feature_path, "r") as f:
            reader = csv.DictReader(f)

            for line in reader:
                concepts.append(line["concept"])
                features.append(line["feature"])
                categories[line["concept"]] = line["category"]

                concept_features[line["concept"]].add(line["feature"])

        concepts = sorted(list(set(concepts)))
        features = sorted(list(set(features)))

        feature_space = defaultdict(lambda: defaultdict(list))

        for c, f in itertools.product(concepts, features):
            if f in concept_features[c]:
                feature_space[f]["positive"].append(c)
            else:
                feature_space[f]["negative"].append(c)

        for _, v in feature_space.items():
            v.default_factory = None

        concept_features.default_factory = None
        feature_space.default_factory = None
        self.feature_lexicon.default_factory = None

        return feature_space, concept_features, features, concepts, categories

    def load_taxonomy(self):
        """
        Builds a taxonomy using WordNet senses of the noun concepts in our database.
        """
        Taxonomy = taxonomy.Nodeset(taxonomy.Node)
        self.lexicon = defaultdict(Concept)

        root = wn.synsets("entity")[0]

        with open(self.concept_path, "r") as f:
            instances = csv.DictReader(f)
            for instance in instances:
                # formatted as category, concept, sensekey, article
                if (
                    instance["concept"] != "kitchen_scales"
                    and instance["category"] != "reading"
                ):
                    try:
                        item = wn.lemma_from_key(instance['sensekey']).synset()
                        self.lexicon[instance["concept"]] = Concept(
                            instance["concept"],
                            instance["category"],
                            item.name(),
                            instance["article"].replace("_", " "),
                        )
                        # self.concepts.append(instance['concept'])
                    except (NameError, ValueError):
                        print(f"Incorrect synset: {instance}")
                    paths = [p for p in item.hypernym_paths() if root in p]

                    if len(paths) <= 1:
                        path = paths[0]
                    else:
                        lens = [len(p) for p in paths]
                        path_idx = list_utils.argmax(lens)
                        path = paths[path_idx]

                    idx = path.index(root)
                    path = path[idx + 1 :]

                    parent = "entity"
                    for synset in path:
                        name = synset.lemma_names()[0].replace("_", " ")
                        name = synset.name()
                        Taxonomy[parent].add_child(Taxonomy[name])
                        parent = name
                    Taxonomy[parent].add_child(Taxonomy[f"{instance['concept']}"])

        Taxonomy.default_factory = None
        self.lexicon.default_factory = None

        return Taxonomy

    def verbalize(self, concept, feature_phrase, templated=False):
        """
        Verbalizes a pair of concept and feature_phrase into natural language sentence.
        """
        if concept in self.concepts:
            article = self.lexicon[concept].article
        else:
            article = engine.a(concept)

        if templated:
            sentence = f"<c> {concept} </c> <p> {feature_phrase} </p>"
        else:
            sentence = f"{article} {feature_phrase}."

        return sentence
