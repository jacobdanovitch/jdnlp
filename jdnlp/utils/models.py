from typing import Dict, List

from allennlp.training.callbacks.callback import Callback, handle_event
from allennlp.training.callbacks.events import Events
from allennlp.training import util as training_util

import neptune

def compare(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


@Callback.register("neptune")
class NeptuneCallback(Callback):
    def __init__(self, project: str, experiment_args: Dict[str, any], artifacts: List[str] = None):
        super().__init__()
        neptune.init(project)
        
        self.experiment_args = experiment_args
        self.artifacts = artifacts
        self.client = neptune.create_experiment(**experiment_args)
    
    def log_metrics(self, metrics: Dict[str, any], desc: str = None):
        for (name, val) in metrics.items():
            if isinstance(val, str):
                self.client.send_text(name, val)
            else:
                self.client.send_metric(name, val)
                
            if desc:
                self.client.send_text("description", desc)
    
    def end_logging(self, trainer):
        self.log_metrics(trainer.metrics)
        for artifact in self.artifacts:
            self.client.send_artifact(artifact)
            
        self.client.stop()
    
    # happens just after allennlp collect_train_metrics
    @handle_event(Events.VALIDATE, priority=-101)
    def log_train_metrics(self, trainer) -> None:
        metrics = trainer.train_metrics
        self.log_metrics(metrics, desc=training_util.description_from_metrics(metrics))
        
    
    # happens just after allennlp collect_val_metrics
    @handle_event(Events.VALIDATE, priority=101)
    def log_val_metrics(self, trainer) -> None:
        metrics = trainer.val_metrics
        self.log_metrics(metrics, desc=training_util.description_from_metrics(metrics))
    
    @handle_event(Events.ERROR)
    def close_client(self, trainer) -> None:
        self.end_logging(trainer)
    
    @handle_event(Events.TRAINING_END, priority=-1000)
    def close_client(self, trainer) -> None:
        self.end_logging(trainer)