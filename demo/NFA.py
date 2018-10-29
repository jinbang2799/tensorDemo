"""
NFA，非确定性有限自动机
"""


# 定义了一个转移规则
class FARule(object):
    # 初始化
    def __init__(self, state, character, next_state):
        self.state = state
        self.character = character
        self.nextState = next_state

    # 通过判断当前的状态和输入和此规则是否相等来决定是否应该使用该种规则
    def applies(self, state, character):
        return self.state == state and self.character == character

    # 获取下一个规则
    def next_state(self):
        return self.nextState


# 同样，我们需要一个规则转移的集合
class FARuleBook(object):
    # 传入一个规则集合
    def __init__(self, rule_set):
        self.ruleSet = rule_set

    # 通过当前的状态集合和输入，根据转移规则集合，计算出所有可能的输出状态
    def get_next_states(self, current_states, character):
        next_states = []
        rule_set = self.ruleSet
        for current_state in current_states:
            for rule in rule_set:
                if rule.applies(current_state, character):
                    next_states.append(rule.next_state())
        # 使用 set 有助于去除重复的集合
        return set(next_states)


# 初始化一个NFA的规则集合
rulebook = FARuleBook([
    FARule(1, 'a', 1),
    FARule(1, 'b', 1),
    FARule(1, 'b', 2),
    FARule(2, 'a', 3),
    FARule(2, 'b', 3),
    FARule(3, 'a', 4),
    FARule(3, 'b', 4)
])


# 非确定性有限自动机
class NFA(object):
    # 初始化包括初始状态、接受状态、规则转移集合
    def __init__(self, current_state, accept_state, rule_book):
        self.current_state = current_state
        self.accept_state = accept_state
        self.rule_book = rule_book

    # 通过判断接受状态是否处于当前状态之中来测试当前是否可能处于接受状态
    def applies(self):
        return self.accept_state in self.current_state

    # 读入一个输入，根据当前状态获取到所有的可能的输出状态的集合
    def read_character(self, character):
        self.current_state = self.rule_book.get_next_states(self.current_state, character)

    # 同理，制作一个方便读入字符串的方法
    def read_string(self, string):
        for c in string:
            self.read_character(c)


# 创建一个NFA，切记输入状态应该是一个集合
nfa = NFA([1], 4, rulebook)
# 读入 bbb
nfa.read_string('bbbaba')
# 判断输入是否可能被接受（即输入是否是该有限自动机的正则语言）
print(nfa.applies())

print(nfa.current_state)
