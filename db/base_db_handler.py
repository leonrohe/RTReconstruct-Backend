from abc import ABC, abstractmethod
from common_utils.myutils import ModelResult

class DBHandler(ABC):
    """
    Abstract base class for database handlers.
    This class defines the interface for all database operations.
    """

    @abstractmethod
    def connect(self):
        """
        Connect to the database.
        This method should be implemented by subclasses to establish a connection.
        """
        pass

    @abstractmethod
    def disconnect(self):
        """
        Disconnect from the database.
        This method should be implemented by subclasses to close the connection.
        """
        pass

    @abstractmethod
    def insert_result(self, model_name: str, result: ModelResult):
        """
        Insert a model result into the database.
        This method should be implemented by subclasses to handle the insertion logic.
        """
        pass

    def __enter__(self):
        self.connect()
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()