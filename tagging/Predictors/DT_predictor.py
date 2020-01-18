from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance


@Predictor.register('DT_predictor')
class CoNLL03Predictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs['texts'] = [str(token) for token in instance.fields['tokens'].tokens]
        outputs['data_id'] = [str(token) for token in instance.fields['data_id'].tokens]

        return sanitize(outputs)