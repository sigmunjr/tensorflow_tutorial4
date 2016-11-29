#ORIGINALLY FROM https://gist.github.com/vinhkhuc/7ec5bf797308279dc587

import random
import numpy as np
import tensorflow as tf

seed_value = 42
tf.set_random_seed(seed_value)
random.seed(seed_value)

def one_hot(v):
  return np.eye(vocab_size)[v]

# Data I/O
vocab_size = 10


# Hyper-parameters
hidden_size   = 100  # hidden layer's size
seq_length    = 25   # number of steps to unroll
learning_rate = 1e-3

inputs_     = tf.placeholder(shape=[None], dtype=tf.int32, name="inputs")
targets_    = tf.placeholder(shape=[None], dtype=tf.int32, name="targets")
inputs = tf.to_float(tf.one_hot(inputs_, vocab_size))
targets = tf.to_float(tf.one_hot(targets_, vocab_size))
init_state = tf.placeholder(shape=[1, hidden_size], dtype=tf.float32, name="state")

initializer = tf.random_normal_initializer(stddev=0.1)

class MyRNN:
  def __init__(self, vocab_size, hidden_size):
    #INITIALIZE YOUR RNN

  def __call__(self, x, hs_t):
    #CALCULATE AND RETURN OUTPUT
    return None, None

rnn = MyRNN(vocab_size, hidden_size)

hs_t = init_state
ys = []
for t, xs_t in enumerate(tf.split(0, seq_length, inputs)):
  ys_t, hs_t = rnn(xs_t, hs_t)
  ys.append(ys_t)


hprev = hs_t
output_softmax = tf.nn.softmax(ys[-1])  # Get softmax for sampling

outputs = tf.concat(0, ys)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, targets))

# Minimizer
minimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grads_and_vars = minimizer.compute_gradients(loss)

# Gradient clipping
grad_clipping = tf.constant(5.0, name="grad_clipping")
clipped_grads_and_vars = []
for grad, var in grads_and_vars:
  clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
  clipped_grads_and_vars.append((clipped_grad, var))

# Gradient updates
updates = minimizer.apply_gradients(clipped_grads_and_vars)

# Session
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Initial values
import Tkinter as tk

class KeyboardListner:
  def __init__(self):
    self.hprev_val = np.zeros([1, hidden_size])
    self.input_buffer = [0]*seq_length
    self.cnt = 0
    self.correct = 0


  def __call__(self, event):
    self.cnt += 1
    input_nr = int(event.char)
    self.new_buffer = self.input_buffer[1:] + [input_nr]
    hprev_val, loss_val, _, out = sess.run([hprev, loss, updates, outputs],
                                      feed_dict={inputs_: self.input_buffer,
                                                 targets_: self.new_buffer,
                                                 init_state: self.hprev_val})
    self.input_buffer = self.new_buffer
    text.tag_config("fail", foreground='red')
    text.tag_config("success", foreground='green')
    pred_nr = out.argmax(1)[-1]
    is_correct = input_nr != pred_nr
    if is_correct:
      self.correct += 1
    tag = ("fail",) if not is_correct else ("success",)
    text.insert('end', 'bot predicted %d\n' % (out.argmax(1)[-1], ), tag)
    root.wm_title("Human: %.2f correct" % (100*self.correct/float(self.cnt)))


root = tk.Tk()
root.geometry('300x200')
text = tk.Text(root, background='black', foreground='white', font=('Comic Sans MS', 12))
text.pack()
keyListner = KeyboardListner()
root.bind('<KeyPress>', keyListner)
root.wm_title('rnn vs. human')
root.mainloop()