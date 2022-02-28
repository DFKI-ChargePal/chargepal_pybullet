""" This file defines the Bullet observer class. """
import abc

class BulletObserver(object):
    """ Bullet observer interface. """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def update_bullet_id(self) -> None:
        """ Update all bullet client connection id's. """
        raise NotImplementedError('Must be implemented in subclass.')
