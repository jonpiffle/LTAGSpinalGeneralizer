import copy
import itertools
import config
from reward import Reward
from default.semantics import SemanticMeaning

class SpinalReward(Reward):
    def __init__(self, worldfile, goalfile):
        self.cached = False
        self.world = set()
        self.entities = set()
        self.goals = set()
        self.relations = {}

        for statement in worldfile:
            s_meaning = SemanticMeaning.parse(statement)
            for argument in s_meaning.meaning_arguments:
                if argument not in self.entities:
                   self.entities.add(argument)
            self.world.add(s_meaning)

        for goal in goalfile:
            self.goals.add(SemanticMeaning.parse(goal))

        for meaning in self.world:
            for argument in meaning.meaning_arguments:
                if argument not in self.relations.keys():
                    self.relations[argument] = []
                self.relations[argument].append(meaning)

    def _evaluate(self, tree):
        total_possible, max_score = 0.0, float('-inf')
        max_binding = {}
        describes_count = {}
        
        entities_in_goals = self.get_entities_in_goals()
        entities, semantics = tree.fol_semantics()
        semantic_meaning = [SemanticMeaning.parse(s) for s in semantics]
        bindings = self.get_possible_bindings(entities)

        for binding, inverse_binding in bindings:
            binding_meaning = self.mutate_meanings_for_new_assignment(binding, semantic_meaning)

            score = 0.0
            if self.assignment_is_possible(binding_meaning):
                score += 500 * sum([1 for g in self.goals if self.fulfills_goal(binding_meaning, g)])
                score += 100 * sum([1 if self.fulfills_goal(binding_meaning, p) else -1 for p in semantic_meaning])
                total_possible += 1

            for entity in self.entities:
                if self.describes(binding_meaning, entity) and entity in inverse_binding:
                    describes_count.setdefault(inverse_binding[entity], set()).add(entity)

            if score > max_score:
                max_score = score
                max_binding = binding

            for entity in entities_in_goals:
                if entity not in describes_count:
                    max_score -= 50
                else:
                    max_score += 50 / len(describes_count[entity])

        final_val = (max_score / (total_possible or 300.0)) - (0.001 * len(str(tree)))
        #print(tree, tree.fol_semantics(), final_val)
        return final_val

    def describes(self, meanings, entity):
        world_entity = set(filter(lambda x: entity in x.meaning_arguments, self.world))
        assignment_entity = set(filter(lambda x: entity in x.meaning_arguments, meanings))
        return assignment_entity <= world_entity

    def get_possible_bindings(self, objects):
        possible_bindings = []
        perm_len = min([len(objects), len(self.entities)])

        if perm_len == 0:
            return possible_bindings

        for entity_perm in itertools.permutations(self.entities, perm_len):
            binding_lists = [((o, e), (e, o)) for e, o in zip(entity_perm, objects)]
            binding, inverse_binding = [dict(l) for l in zip(*binding_lists)]
            possible_bindings.append((binding, inverse_binding))

        return possible_bindings

    def mutate_meanings_for_new_assignment(self, assignment_map, semantic_meanings):
        new_meanings = set()
        for meaning in semantic_meanings:
            try:
                meaning_arguments = [assignment_map[arg] for arg in meaning.meaning_arguments]
            except KeyError as e:
                continue
            new_meanings.add(SemanticMeaning(meaning.meaning_string, meaning_arguments))
        return new_meanings

    def get_entities_in_goals(self):
        entity_set = set()
        for goal in self.goals:
            entity_set.update(goal.meaning_arguments)
        return entity_set

    def assignment_is_possible(self, meanings):
        meaning_set = set(meanings)
        return meaning_set <= self.world

    def fulfills_goal(self, meanings, goal):
        return goal in set(meanings)
