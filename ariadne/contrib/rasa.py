

from rasa.cli.utils import get_validated_path
from rasa.model import get_latest_model, get_model_subdirectories, unpack_model
from rasa.nlu.model import Interpreter
from pathlib import Path
from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE, SENTENCE_TYPE
from ariadne.protocol import TrainingDocument
from cassis import Cas
from typing import List, Optional, Any
import logging 
logger = logging.getLogger(__name__)


# this model is user independent
# the user_id which is inherited from the Classifier is 
class RasaClassifier(Classifier):
    model = None
    model_directory = None 
    def __init__(self, model_directory: Path = None):
        self.model_directory = model_directory
        self.model = self._load_model("user_id") # don't need a user_id here -> I just do it because of inheritance
        super().__init__(model_directory)
        
    def fit(self, documents: List[TrainingDocument], layer: str, feature: str, project_id, user_id: str):
        raise NotImplementedError("Fitting the rasa client is not implemented yet")

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        if self.model is None:
            logger.debug("No trained model ready yet!")
            return
        
        for sentence in cas.select(SENTENCE_TYPE):
            predicted = self.model.parse(sentence.get_covered_text())["intent"]["name"]
            prediction = create_prediction(cas, layer, feature, sentence.begin, sentence.end, predicted)
            cas.add_annotation(prediction)

    def _load_model(self, user_id: str) -> Optional[Any]:
            """
            This loads the Rasa NLU interpreter. It is able to apply all NLU
            pipeline steps to a text that you provide it.
            """
            model_path = self._get_model_path(user_id)
            # model = get_validated_path(model_path, "model")
            
            # _, nlu_model = get_model_subdirectories(model_path)
            return Interpreter.load(model_path)

    def _save_model(self, user_id: str, model: Any):
        return super()._save_model(user_id, model)
    
    def _get_model_path(self, user_id: str) -> Path:
        return unpack_model(get_latest_model(self.model_directory))+ '/nlu' # just for testing purpose



    

