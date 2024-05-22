# -*- coding: utf-8 -*-
import torch.nn as nn


class EvalPropMetric(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.prop = args.eval_prop
        self.metric = args.eval_metric

        self._set_loss_fn()

    def _set_loss_fn(self):
        if self.metric == "mae":
            self.loss_fn = nn.L1Loss(reduction="mean")
        elif self.metric == "mse":
            self.loss_fn = nn.MSELoss(reduction="mean")
        else:
            raise NotImplementedError(
                f"Evaluation metric {self.metric} not implemented."
            )

    def forward(self, model_output, batch_data):
        y_pred = model_output[self.prop]
        y_true = batch_data[self.prop]

        loss = self.loss_fn(y_pred, y_true)

        logging_output = {
            "total_loss": loss,
        }

        return loss, logging_output
