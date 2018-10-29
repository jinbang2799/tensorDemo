"""
DFA，确定性有限自动机
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


# 如果上面的类是一个转移规则，那么这里就是存储了一个有限自动机的转移规则的集合
class FARuleBook(object):
    def __init__(self, rule_set):
        self.ruleSet = rule_set

    def rules(self):
        return self.ruleSet


# 初始化一个有限自动机的规则集合
rulebook = FARuleBook([
    FARule(1, 'a', 2),
    FARule(1, 'b', 1),
    FARule(2, 'a', 2),
    FARule(2, 'b', 1)
])


# 定义一个确定性有限自动机
class DFA(object):
    # 初始化有限自动机时包含了初始状态、接受状态和转移规则的集合
    def __init__(self, current_state, accept_state, rule_book):
        self.currentState = current_state
        self.acceptState = accept_state
        self.ruleBook = rule_book.rules()

    # 当输入一个字符时，根据当前状态和输入去转移规则中寻找对应的转移规则，
    # 根据规则获取下一个状态，并把下一个状态置为当前状态
    def input_character(self, character):
        for r in self.ruleBook:
            if r.applies(self.currentState, character):
                self.currentState = r.next_state()

    # 根据当前状态来判断是否是输入状态
    def is_accept(self):
        return self.currentState == self.acceptState

    # 为了方便操作，同时也提供了一个可以输入字符串的方法
    def input_string(self, string):
        for c in string:
            self.input_character(c)


# 初始化一个DFA
dfa = DFA(1, 2, rulebook)
# 读入字符串
dfa.input_string('aaaaaaaaaaaaaaaaaaab')
# 判断当前是否处于接受状态
print(dfa.is_accept())

print(dfa.currentState)
