# -*- coding: UTF8 -*-
import os
import math
import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid as sig
from theano.tensor.shared_randomstreams import RandomStreams

import scipy.sparse as sp
from theano import sparse

#Don't use a python long as this don't work on 32 bits computers.
#numpy.random.seed(0xbeef)

rng = RandomStreams(seed=numpy.random.randint(1 << 30))
#theano.config.warn.subtensor_merge_bug = False
#theano.config.compute_test_value = 'raise'
theano.config.exception_verbosity='high'
#theano.config.optimizer='None'
numpy.seterr(all='warn')

def shared_normal(num_rows, num_cols, scale=1, name=None):
    '''Initialize a matrix shared variable with normally distributed elements.'''
    return theano.shared(numpy.random.normal(scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX), name=name)

def shared_zeros(*shape):
    '''Initialize a vector shared variable with zero elements.'''
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))

class LSTM:
    # Sig is 0/1, Tanh is -1/1

    def __init__(self, n_input=3, n_memblock=100, n_output=2, lr=0.0001, m=0.9, l2rate=0.0001, dense=True):
        self.dense = dense
        input_sequence = T.matrix()
        gold_sequence = T.matrix() # 1, n_output
        
        #input_sequence.tag.test_value = [[0,0,1],[0,1,0],[1,0,0]]
        #gold_sequence.tag.test_value = [[1,0],[0,1],[0,0]]
        
        ''' START WEIGHTS - 0=forward; 1=backward'''
        wiig = shared_normal(n_input, n_memblock, 0.01,"wiig0"),shared_normal(n_input, n_memblock, 0.01,"wiig1") # Weights from inputs to gates
        wmig = shared_normal(n_memblock, n_memblock, 0.01,"wmig0"),shared_normal(n_memblock, n_memblock, 0.01,"wmig1") # Weights from cells to gates - peepholes
        #big = shared_zeros(n_memblock,"big0"),shared_zeros(n_memblock,"big1")
        big = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"big0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"big1")
        
        wifg = shared_normal(n_input, n_memblock, 0.01,"wifg0"),shared_normal(n_input, n_memblock, 0.01,"wifg1")
        wmfg = shared_normal(n_memblock, n_memblock, 0.01,"wmfg0"),shared_normal(n_memblock, n_memblock, 0.01,"wmfg1")
        #bfg = shared_zeros(n_memblock,"bfg0"),shared_zeros(n_memblock,"bfg1")
        bfg = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"bfg0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"bfg1")
        
        wiog = shared_normal(n_input, n_memblock, 0.01,"wiog0"),shared_normal(n_input, n_memblock, 0.01,"wiog1")
        wmog = shared_normal(n_memblock, n_memblock, 0.01,"wmog0"),shared_normal(n_memblock, n_memblock, 0.01,"wmog1")
        #bog = shared_zeros(n_memblock,"bog0"),shared_zeros(n_memblock,"bog1")
        bog = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"bog0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"bog1")
        
        wim = shared_normal(n_input, n_memblock, 0.01,"wim0"),shared_normal(n_input, n_memblock, 0.01,"wim1") # Weight from input to mem
        #bm = shared_zeros(n_memblock,"bm0"),shared_zeros(n_memblock,"bm1") # Bias from input to mem
        bm = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"bm0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"bm1")
        
        wmo = shared_normal(n_memblock, n_output, 0.01,"wmo0"),shared_normal(n_memblock, n_output, 0.01,"wmo1") # Weight from input to mem
        
        slo = theano.shared(numpy.random.normal(scale = 0.01), name="slo0"), theano.shared(numpy.random.normal(scale = 0.01), name="slo1")
        bo = theano.shared(numpy.zeros(n_output, dtype=theano.config.floatX),"bo") # Bias from input to mem
        ''' END OF WEIGHTS '''
        
        self.params = wiig[0], wiig[1], big[0], big[1], wifg[0], wifg[1], bfg[0], bfg[1], wiog[0], wiog[1], bog[0], bog[1], wmig[0], wmig[1], wmfg[0], wmfg[1], wmog[0], wmog[1], wim[0], wim[1], bm[0], bm[1], wmo[0], wmo[1], slo[0], slo[1], bo
        
        ''' START DELTAS - 0=forward; 1=backward'''
        dwiig = shared_normal(n_input, n_memblock, 0.01,"dwiig0"),shared_normal(n_input, n_memblock, 0.01,"dwiig1") # Weights from inputs to gates
        dwmig = shared_normal(n_memblock, n_memblock, 0.01,"dwmig0"),shared_normal(n_memblock, n_memblock, 0.01,"dwmig1") # Weights from cells to gates - peepholes
        #dbig = shared_zeros(n_memblock,"big0"),shared_zeros(n_memblock,"dbig1")
        dbig = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbig0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbig1")
        
        dwifg = shared_normal(n_input, n_memblock, 0.01,"dwifg0"),shared_normal(n_input, n_memblock, 0.01,"dwifg1")
        dwmfg = shared_normal(n_memblock, n_memblock, 0.01,"dwmfg0"),shared_normal(n_memblock, n_memblock, 0.01,"dwmfg1")
        #dbfg = shared_zeros(n_memblock,"bfg0"),shared_zeros(n_memblock,"dbfg1")
        dbfg = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbfg0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbfg1")
        
        dwiog = shared_normal(n_input, n_memblock, 0.01,"dwiog0"),shared_normal(n_input, n_memblock, 0.01,"dwiog1")
        dwmog = shared_normal(n_memblock, n_memblock, 0.01,"dwmog0"),shared_normal(n_memblock, n_memblock, 0.01,"dwmog1")
        #dbog = shared_zeros(n_memblock,"bog0"),shared_zeros(n_memblock,"dbog1")
        dbog = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbog0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbog1")
        
        dwim = shared_normal(n_input, n_memblock, 0.01,"dwim0"),shared_normal(n_input, n_memblock, 0.01,"dwim1") # Weight from input to mem
        #dbm = shared_zeros(n_memblock,"bm0"),shared_zeros(n_memblock,"dbm1") # Bias from input to mem
        dbm = theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbm0"),theano.shared(numpy.zeros(n_memblock, dtype=theano.config.floatX),"dbm1")
        
        dwmo = shared_normal(n_memblock, n_output, 0.01,"dwmo0"),shared_normal(n_memblock, n_output, 0.01,"dwmo1") # Weight from input to mem
        
        dslo = theano.shared(numpy.random.normal(scale = 0.01), name="dslo0"), theano.shared(numpy.random.normal(scale = 0.01), name="dslo1")
        
        dbo = theano.shared(numpy.zeros(n_output, dtype=theano.config.floatX),"dbo") # Bias from input to mem
        ''' END OF DELTAS '''
        
        self.deltas = dwiig[0], dwiig[1], dbig[0], dbig[1], dwifg[0], dwifg[1], dbfg[0], dbfg[1], dwiog[0], dwiog[1], dbog[0], dbog[1], dwmig[0], dwmig[1], dwmfg[0], dwmfg[1], dwmog[0], dwmog[1], dwim[0], dwim[1], dbm[0], dbm[1], dwmo[0], dwmo[1], dslo[0], dslo[1], dbo
        
        init_mem = shared_zeros(n_memblock)
        
        # EXPRESSIONS - Forward
        def recurrence(input, pmem, i):
            i = i.value
            ingate   = sig(T.dot(input, wiig[i]) + T.dot(pmem, wmig[i]) + big[i])
            forgate  = sig(T.dot(input, wifg[i]) + T.dot(pmem, wmfg[i]) + bfg[i])
            #mem      = forgate * pmem            + ingate * T.tanh(T.dot(input, wim[i]) + bm[i]) # Use sig or tan???
            mem      = T.tanh(forgate * pmem + ingate * T.tanh(T.dot(input, wim[i]) + bm[i])) # instead of identity, use tanh for mem out
            outgate  = sig(T.dot(input, wiog[i]) + T.dot(mem, wmog[i])  + bog[i])
            layerout = T.tanh(T.dot(outgate * mem, wmo[i]))
            #print layerout.shape.eval()
            return mem, layerout
        
        #Forward Pass
        (_, output_sequencef), updf = theano.scan(fn=recurrence, sequences = input_sequence, non_sequences = 0, outputs_info = [init_mem, None])
        (_, output_sequencebp), updb = theano.scan(fn=recurrence, sequences = input_sequence, non_sequences = 1, outputs_info = [init_mem, None], go_backwards=True)
        output_sequenceb = output_sequencebp[::-1]
        
        presig_output_sequence, train_updates = theano.scan(fn=lambda x, y: (x*slo[0]+y*slo[1]+bo), sequences = [output_sequencef, output_sequenceb], outputs_info=[None])
        
        # avoid log(0) for log(scan(sigmoid()))
        output_sequence = sig(presig_output_sequence)
        # output_sequence become a batch of output vectors
        train_updates.update(updf)
        train_updates.update(updb)
        
        l2 = 0
        for p in self.params:
            l2 += T.sum(p*p)
        
        # Loss Function
        outloss = T.nnet.binary_crossentropy(output_sequence, gold_sequence).mean() + l2*l2rate # TODO: check if the dimensions match here
        # consider using multi-category? because binary allows multiple 1's in the vector
    
        # Backward Pass
        gradient = T.grad(outloss, self.params, consider_constant=[input_sequence, gold_sequence])
        
        train_updates.update(((p, p + m * d - lr * g) for p, g, d in zip(self.params, gradient, self.deltas)))
        train_updates.update(((d, m * d - lr * g) for p, g, d in zip(self.params, gradient, self.deltas)))
        
        target = T.iround(gold_sequence)
        output = T.iround(output_sequence)
        tp = T.sum(T.and_(target,output))
        p = tp/(T.sum(target))
        r = tp/(T.sum(output))
        f = ( 2 * p * r )/(p+r)
        
        ct = T.sum(target)
        co = T.sum(output)
    
        #self.train_function = theano.function([input_sequence,gold_sequence], [output_sequence], updates=train_updates)
        self.train_function = theano.function([input_sequence,gold_sequence], [], updates=train_updates)
        #self.validate_function = theano.function([input_sequence,gold_sequence], [outloss,output_sequence])
        self.test_function = theano.function([input_sequence,gold_sequence], [outloss, ct, co, tp])
        self.generate_function = theano.function([input_sequence], output)
    
    
    def train(self, data):
        #dataset = [([[0,0,1],[0,1,0],[1,0,0]],[[1,0],[0,1],[0,0]]),([[0,0,0],[0,1,1],[1,0,0]],[[1,0],[1,1],[0,0]])]
        for ip, gold in data:
            if not self.dense: ip, gold = ip.todense(), gold.todense()
            self.train_function(ip, gold)
        return

    def test(self,data):
        act = 0.0
        aco = 0.0
        atp = 0.0
        costs = []
        for ip, gold in data:
            if not self.dense: ip, gold = ip.todense(), gold.todense()
            cost, ct, co, tp = self.test_function(ip, gold)
            costs.append(cost)
            act = act + ct
            aco = aco + co
            atp = atp + tp
        
        return numpy.mean(costs), act, aco, atp, atp/aco if aco else 0, atp/act if act else 0, 2*atp/(aco+act) if (aco+act) else 0
    
    def generate(self, data):
        ops = []
        for ip, gold in data:
            if not self.dense: ip, gold = ip.todense(), gold.todense()
            op = self.generate_function(ip)
            ops.append(op)
        return ops

    def save(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        for param in self.params:
            numpy.save(os.path.join(folder, param.name + '.npy'), param.get_value())
        '''
        for delta in self.deltas:
            numpy.save(os.path.join(folder, delta.name + '.npy'), delta.get_value())
        '''    
    
    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder, param.name + '.npy')))
        '''
        for delta in self.deltas:
            delta.set_value(numpy.load(os.path.join(folder, delta.name + '.npy')))
        '''




