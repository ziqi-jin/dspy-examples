from ast import main
import dspy          
from get_dataset import custom_trainset as train_set          
from get_dataset import custom_testset as test_set

from dspy.datasets.gsm8k import gsm8k_metric          
from dspy.teleprompt import BootstrapFewShot

if __name__  == '__main__':
        
    #   定义并设置大模型          
    model_name = 'llama3'          
    lm = dspy.OllamaLocal(model=model_name)          
    dspy.settings.configure(lm=lm)

    
    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")

        def forward(self, question):
            return self.prog(question=question)

    cot = CoT()
    
    config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=2)

    #   Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
    teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)

        
    import langwatch          
    langwatch.login()          
    langwatch.dspy.init(experiment="test", optimizer=teleprompter)          
    optimized_cot = teleprompter.compile(cot, trainset=train_set)
    