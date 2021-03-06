#!/usr/bin/env python

class DAO(object):
    """
    I'm not actually sure how abstractions/interfaces are implemented in Python. But here you go.
    It's supposed to describe the interface of how input--output pairs are retrieved.
    """

    def data_dyad(self):
        raise NotImplementedError()

    def data_monad(self):
        raise NotImplementedError()

    def iter(self):
        # should return an iterable that loops through "all" data in some sense.
        raise NotImplementedError()