import dspy
from get_dataset import custom_trainset as train_set
from get_dataset import custom_testset as test_set
from dspy.datasets.gsm8k import gsm8k_metric
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate import Evaluate


PROMPT_SWITCH = True

if __name__ == '__main__':

    #  定义并设置大模型
    model_name = 'llama3'
    lm = dspy.OllamaLocal(model=model_name)
    dspy.settings.configure(lm=lm) 

    print(f"## 创建并设置大模型 {model_name} ##")
    print(f"\n## 对模板和参数进行调优 - start ##\n")

    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)


    cot = CoT()
    evaluate = Evaluate(devset=test_set, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
    evaluate(cot) 

    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)

    #  Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
    #  可以调整 train_set 长度
    optimized_cot = teleprompter.compile(cot, trainset=train_set)
    optimized_cot.save("./test.json")
    question = "3+3+5=?"
    response = optimized_cot(question=question)
    print(f"问题：{question} \n答案：{response.answer}")
    print("prompt: ")
    lm.inspect_history(n=1) if PROMPT_SWITCH else print("if you want see prompt, set PROMPT_SWITCH = True")
    print(f"\n## 对模板和参数进行调优 - end ##\n")
    evaluate(optimized_cot)
