
SUCCESS = 1
FAILURE = 0
RUNNING = 2

class BehaviorNode:
    def tick(self):
        raise NotImplementedError("tick() must be implemented by subclass.")


class SequenceNode(BehaviorNode):
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status != SUCCESS:
                return status
        return SUCCESS


class SelectorNode(BehaviorNode):
    def __init__(self, children):
        self.children = children

    def tick(self):
        for child in self.children:
            status = child.tick()
            if status == SUCCESS:
                return SUCCESS
        return FAILURE


class ConditionNode(BehaviorNode):
    def __init__(self, condition_func):
        """
        condition_func should be a function with no arguments that returns True or False.
        """
        self.condition_func = condition_func

    def tick(self):
        if self.condition_func():
            return SUCCESS
        else:
            return FAILURE


class ActionNode(BehaviorNode):
    def __init__(self, action_func):
        """
        action_func should be a function with no arguments that performs an action 
        and returns SUCCESS or FAILURE.
        """
        self.action_func = action_func

    def tick(self):
        return self.action_func()