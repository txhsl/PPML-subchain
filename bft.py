import time
import _thread
from queue import Queue
from enum import Enum

class State(Enum):
    WAITING = 0
    INITIAL = 1
    PRIMARY = 2
    BACKUP = 3
    PREPARED = 4
    COMMITED = 5
    VIEWCHANGING = 6

class MessageType(Enum):
    PREPAREREQUEST = 0
    PREPARERESPONSE = 1
    COMMIT = 2
    VIEWCHANGE = 3

class Message:
    def __init__(self, msg_type, height, view, model, tx_heads):
        self.msg_type = msg_type
        self.height = height
        self.view = view
        self.model = model
        self.tx_heads = tx_heads

class Node:
    def __init__(self, id, bft_size):
        self.id = id
        self.height = 0
        self.view = 0
        self.state = State.WAITING
        self.state_lock = _thread.allocate_lock()

        self.bft_size = bft_size
        self.vote_counter = 0
        self.change_counter = 0

        self.tx_limit = 16

        self.peers = []
        self.msgs = Queue()

        self.model = None
        self.txs = []
    def is_primary(self):
        return (self.height + self.view) % self.bft_size == self.id
    def have_enough_vote(self):
        return self.vote_counter >= 2 * self.bft_size // 3 + 1
    def broadcast(self, msg):
        for peer in self.peers:
            if peer.id != self.id:
                peer.msgs.put(msg)
    def propose(self, model, tx_heads):
        self.broadcast(Message(MessageType.PREPAREREQUEST, self.height, self.view, model, tx_heads))
    def prepare(self, msg):
        msg = Message(MessageType.PREPARERESPONSE, msg.height, msg.view, msg.model, msg.tx_heads)
        self.broadcast(msg)
    def commit(self, msg):
        msg = Message(MessageType.COMMIT, msg.height, msg.view, msg.model, msg.tx_heads)
        self.broadcast(msg)
    def change_view(self):
        self.broadcast(Message(MessageType.PREPAREREQUEST, self.height, self.view, None, []))

    def run(self):
        if self.state == State.WAITING:
            self.vote_counter = 0
            self.state = State.INITIAL

            if self.is_primary():
                self.state = State.PRIMARY
                self.vote_counter += 1
                self.propose(None, [])
            else:
                self.state = State.BACKUP
        else:
            if self.state == State.COMMITED:
                print("[Node ", self.id, "] Consensus success. Height: ", self.height, ", view: ", self.view)
                self.view = 0
                self.height += 1
                self.state = State.WAITING
            else:
                #print("[Node ", self.id, "] Consensus failed, view changing. Height: ", self.height, ", view: ", self.view)
                #self.view += 1
                #print("[Node ", self.id, "] View changed, number: ", self.view)
                while self.msgs.qsize() > 0:
                    msg = self.msgs.get()
                    if msg.msg_type == MessageType.PREPAREREQUEST:
                        if self.state == State.BACKUP:
                            if msg.height != self.height or msg.view != self.view:
                                continue
                            self.vote_counter += 1
                            self.prepare(msg)
                            self.vote_counter += 1
                            self.signal = True
                    elif msg.msg_type == MessageType.PREPARERESPONSE:
                        if self.state == State.PRIMARY or self.state == State.BACKUP:
                            if msg.height != self.height or msg.view != self.view:
                                continue
                            self.vote_counter += 1
                            if self.have_enough_vote():
                                self.state = State.PREPARED
                                self.commit(msg)
                                self.vote_counter = 0
                                self.signal = True
                    elif msg.msg_type == MessageType.COMMIT:
                        if self.state == State.PREPARED:
                            if msg.height != self.height or msg.view != self.view:
                                continue
                            self.vote_counter += 1
                            if self.have_enough_vote:
                                self.state = State.COMMITED
                                print("[Node ", self.id, "] Commits. But nothing is done here")
                    elif msg.msg_type == MessageType.VIEWCHANGE:
                        if self.state != State.COMMITED and self.state != State.INITIAL:
                            if msg.height != self.height or msg.view != self.view + 1:
                                continue
                            self.change_counter += 1
                            if self.have_enough_vote:
                                self.state = State.VIEWCHANGING