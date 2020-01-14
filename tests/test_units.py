import unittest
import numpy as np
from unittest.mock import MagicMock
from colna.lossconversion import dBcm_to_loss_per_m, loss_per_m_to_dBcm, attenuation_to_dBcm, \
    attenuation_to_loss_per_meter, loss_per_meter_to_attenuation, dBcm_to_attenuation
from colna.analyticnetwork import Network, Edge, SymNum, Testbench


class TestUnitconverter(unittest.TestCase):

    def test_loss_conversion(self):
        self.assertEqual(dBcm_to_loss_per_m(1), 100 / (10 * np.log10(np.exp(1))))
        self.assertEqual(attenuation_to_dBcm(1, 1), 0)

        self.assertEqual(loss_per_m_to_dBcm(dBcm_to_loss_per_m(1)), 1)
        self.assertEqual(dBcm_to_attenuation(attenuation_to_dBcm(1, 2), 2), 1)
        self.assertEqual(loss_per_meter_to_attenuation(attenuation_to_loss_per_meter(1, 2), 2), 1)


class TestEdge(unittest.TestCase):

    def setUp(self):
        self.edge = Edge(start='a', end='b', phase=.5, attenuation=.6, delay=0.7)

    def test_edge_initialization(self):
        self.assertEqual(self.edge.start, 'a')
        self.assertEqual(self.edge.end, 'b')
        self.assertEqual(self.edge.phase, 0.5)
        self.assertEqual(self.edge.attenuation, 0.6)
        self.assertEqual(self.edge.delay, 0.7)


class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.net = Network()
        self.net.add_node('a')
        self.net.add_node('b')

    def test_add_node(self):
        # check that nodes where added correctly
        self.assertEqual(self.net.nodes, ['a', 'b'])

        # add a node
        self.net.add_node('d')
        self.assertEqual(self.net.nodes, ['a', 'b', 'd'])

    def test_add_node_duplication(self):
        """ Tests if node duplication is prohibited"""
        with self.assertRaises(ValueError):
            self.net.add_node('a')

        with self.assertRaises(ValueError):
            self.net.add_node('b')

    def test_add_edge(self):
        """ Tests if edges can be added to the network"""
        test_edge = Edge('a', 'b')
        self.net.add_edge(test_edge)
        self.assertEqual(self.net.edges[0], test_edge)

    def test_add_edge_Errors(self):
        """ Tests edge errors"""
        test_edge = Edge('a', 'c')
        with self.assertRaises(ValueError):
            self.net.add_edge(test_edge)

        test_edge = Edge('d', 'a')
        with self.assertRaises(ValueError):
            self.net.add_edge(test_edge)

        test_edge = Edge('d', 'c')
        with self.assertRaises(ValueError):
            self.net.add_edge(test_edge)

    def test_add_input(self):
        # add inputs to node a
        self.net.add_input('a', amplitude=0.2, phase=0.1, delay=0.1)
        self.assertEqual(self.net.inputs, [(0.2, 0.1, 0.1, 'a')])

        # add input to node b
        self.net.add_input('b', amplitude=0.5, phase=0.2, delay=0.0)
        self.assertEqual(self.net.inputs, [(0.2, 0.1, 0.1, 'a'), (0.5, 0.2, 0.0, 'b')])

        # add second input to node a
        self.net.add_input('a', amplitude=1.8, phase=-0.1, delay=-0.1)
        self.assertEqual(self.net.inputs, [(0.2, 0.1, 0.1, 'a'), (0.5, 0.2, 0.0, 'b'), (1.8, -0.1, -0.1, 'a')])

    def test_add_input_errors(self):
        # add input at non-existing node
        with self.assertRaises(ValueError):
            self.net.add_input('d', amplitude=1, phase=0.1, delay=2)

    def test_evaluate_feed_forward(self):
        """ creates and evaluates a feed forward network """
        ff_net = Network()
        ff_net.add_node('a')
        ff_net.add_node('b')
        ff_net.add_node('c')

        ff_net.add_edge(Edge('a', 'b', phase=0.5, attenuation=0.8, delay=2))
        ff_net.add_edge(Edge('b', 'c', phase=-5, attenuation=1.5, delay=-1))

        ff_net.add_input('a', amplitude=1)

        ff_net.evaluate()

        self.assertEqual(ff_net.nodes_to_output, {'a': [(1, 0, 0, '-a')],
                                                  'b': [(1 * 0.8, 0 + 0.5, 2, '-a-b')],
                                                  'c': [(1 * 0.8 * 1.5, 0 + 0.5 - 5, 2 - 1, '-a-b-c')]})

    def test_evaluate_loop(self):
        edge_1 = Edge('a', 'b', phase=1, attenuation=0.4, delay=2)
        edge_2 = Edge('b', 'c', phase=2, attenuation=0.3, delay=1)
        edge_3 = Edge('c', 'a', phase=3, attenuation=0.2, delay=0)

        expected_result = {'a': [(1, 0.0, 0.0, '-a'), (edge_1.attenuation * edge_2.attenuation * edge_3.attenuation,
                                                       edge_1.phase + edge_2.phase + edge_3.phase,
                                                       edge_1.delay + edge_2.delay + edge_3.delay, '-a-b-c-a')],
                           'b': [(edge_1.attenuation, edge_1.phase, edge_1.delay, '-a-b'),
                                 (edge_1.attenuation * edge_2.attenuation * edge_3.attenuation * edge_1.attenuation,
                                  edge_1.phase + edge_2.phase + edge_3.phase + edge_1.phase,
                                  edge_1.delay + edge_2.delay + edge_3.delay + edge_1.delay, '-a-b-c-a-b')],
                           'c': [(edge_1.attenuation * edge_2.attenuation, edge_1.phase + edge_2.phase,
                                  edge_1.delay + edge_2.delay, '-a-b-c'),
                                 (
                                     edge_1.attenuation * edge_2.attenuation * edge_3.attenuation * edge_1.attenuation * edge_2.attenuation,
                                     edge_1.phase + edge_2.phase + edge_3.phase + edge_1.phase + edge_2.phase,
                                     edge_1.delay + edge_2.delay + edge_3.delay + edge_1.delay + edge_2.delay,
                                     '-a-b-c-a-b-c')]}

        loop_net = Network()
        loop_net.add_node('a')
        loop_net.add_node('b')
        loop_net.add_node('c')
        loop_net.add_edge(edge_1)
        loop_net.add_edge(edge_2)
        loop_net.add_edge(edge_3)
        loop_net.add_input('a', amplitude=1)

        loop_net.evaluate(amplitude_cutoff=1e-3)
        self.assertEqual(loop_net.nodes_to_output, expected_result)

    def test_evaluate_loop_edge_order_independance(self):
        """
        Checks that the order in which we add edge to a network does not matter
        """
        pass

    def test_evaluate_splitting(self):
        """ creates and evaluates a feed forward split """
        edge_1 = Edge('a', 'b', phase=0.5, attenuation=0.5, delay=2)
        edge_2 = Edge('a', 'c', phase=-0.5, attenuation=1.5, delay=-1)

        expected_nodes_to_output = {'a': [(1, 0, 0, '-a')],
                                    'b': [(edge_1.attenuation, edge_1.phase, edge_1.delay, '-a-b')],
                                    'c': [(edge_2.attenuation, edge_2.phase, edge_2.delay, '-a-c')]}

        split_net = Network()
        split_net.add_node('a')
        split_net.add_node('b')
        split_net.add_node('c')

        split_net.add_edge(edge_1)
        split_net.add_edge(edge_2)
        split_net.add_input('a', amplitude=1)
        split_net.evaluate()

        self.assertEqual(split_net.nodes_to_output, expected_nodes_to_output)

    def test_evaluate_combining(self):
        """ creates and evaluates a feed forward combiner """
        edge_1 = Edge('b', 'a', phase=0.5, attenuation=0.5, delay=2)
        edge_2 = Edge('c', 'a', phase=-0.5, attenuation=1.5, delay=-1)

        expected_nodes_to_output = {'a': [(edge_1.attenuation, edge_1.phase, edge_1.delay, '-b-a'),
                                          (edge_2.attenuation, edge_2.phase, edge_2.delay, '-c-a')],
                                    'b': [(1, 0, 0, '-b')],
                                    'c': [(1, 0, 0, '-c')]}

        split_net = Network()
        split_net.add_node('a')
        split_net.add_node('b')
        split_net.add_node('c')

        split_net.add_edge(edge_1)
        split_net.add_edge(edge_2)
        split_net.add_input('b', amplitude=1)
        split_net.add_input('c', amplitude=1)
        split_net.evaluate()

        self.assertEqual(split_net.nodes_to_output, expected_nodes_to_output)


class TestSymNum(unittest.TestCase):

    def setUp(self):
        self.mult_0 = SymNum('mult_0', default=0, product=True)
        self.mult_1 = SymNum('mult_1', default=1, product=True)
        self.mult_5 = SymNum('mult_5', default=5, product=True)
        self.mult_6 = SymNum('mult_6', default=6, product=True)

        self.add_0 = SymNum('add_0', default=0, product=False)
        self.add_1 = SymNum('add_1', default=1, product=False)
        self.add_5 = SymNum('add_5', default=5, product=False)
        self.add_6 = SymNum('add_6', default=6, product=False)

    def test_equal(self):
        add_0_2 = SymNum('add_0', default=0, product=False)
        self.assertEqual(self.add_0 == add_0_2, True)
        add_0_2b = self.add_0
        self.assertEqual(self.add_0 == add_0_2b, True)

    def test_not_equal(self):
        self.assertEqual(self.add_0 == self.add_1, False)
        add_0_2 = SymNum('add_0', default=0.1, product=False)
        self.assertEqual(self.add_0 == add_0_2, False)
        add_0_2 = SymNum('add_01', default=0, product=False)
        self.assertEqual(self.add_0 == add_0_2, False)
        add_0_2 = SymNum('add_0', default=0, product=True)
        self.assertEqual(self.add_0 == add_0_2, False)
        add_0_2 = SymNum('add_0', default=0, product=False, numerical=3)
        self.assertEqual(self.add_0 == add_0_2, False)

    def test_copy(self):
        """ we expect deepcopy of defaults and symbolic dictionary"""

        copy_0 = self.mult_0.__copy__()
        self.assertEqual(
            copy_0.symbolic == self.mult_0.symbolic and copy_0.defaults == self.mult_0.defaults and copy_0.product == self.mult_0.product and copy_0.numerical == self.mult_0.numerical and id(
                copy_0.symbolic) != id(self.mult_0.symbolic) and id(copy_0.defaults) != id(self.mult_0.symbolic), True)

        copy_1 = self.add_1.__copy__()
        self.assertEqual(
            copy_1.symbolic == self.add_1.symbolic and copy_1.defaults == self.add_1.defaults and copy_1.product == self.add_1.product and copy_1.numerical == self.add_1.numerical and id(
                copy_1.symbolic) != id(self.add_1.symbolic) and id(copy_1.defaults) != id(self.add_1.symbolic), True)

    def test_add_mult(self):
        """ Tests if adding multiplicative numbers does raise a Value error"""

        with self.assertRaises(ValueError):
            self.mult_0 + self.mult_1

        with self.assertRaises(ValueError):
            self.add_0 + self.mult_1

        with self.assertRaises(ValueError):
            self.mult_1 + self.add_0

    def test_add(self):
        """ Tests if adding two symnums returns the expected new symnum"""
        add_0 = self.add_1
        add_1 = self.add_5

        sum =   add_0 + add_1
        self.assertEqual(sum.product, False)
        self.assertEqual(sum.shared_default, max(add_0.shared_default, add_1.shared_default))
        self.assertEqual(sum.numerical, 0.0)
        self.assertEqual(sum.symbolic, {'add_5': 1, 'add_1': 1})
        self.assertEqual(sum.defaults, {'add_5': 5, 'add_1': 1})

    def test_mult_add(self):
        """ Tests if multypling additive numbers does raise a Value error"""
        with self.assertRaises(ValueError):
            self.add_0 * self.add_1

        with self.assertRaises(ValueError):
            self.add_0 * self.mult_1

        with self.assertRaises(ValueError):
            self.mult_1 * self.add_0

    def test_mult(self):
        mult_0 = self.mult_1
        mult_1 = self.mult_5

        mult =   mult_0*mult_1
        self.assertEqual(mult.product, True)
        self.assertEqual(mult.shared_default, max(mult_0.shared_default, mult_1.shared_default))
        self.assertEqual(mult.numerical, 1.0)
        self.assertEqual(mult.symbolic, {'mult_1': 1, 'mult_5': 1})
        self.assertEqual(mult.defaults, {'mult_5': 5, 'mult_1': 1})

        mult_2 = mult_1 * 2
        self.assertEqual(mult_2.product, True)
        self.assertEqual(mult_2.shared_default, mult_1.shared_default)
        self.assertEqual(mult_2.numerical, 2.0)
        self.assertEqual(mult_2.symbolic, {'mult_5': 1})
        self.assertEqual(mult_2.defaults, {'mult_5': 5})



    def test_div_add(self):
        """ Tests if multypling additive numbers does raise a Value error"""
        with self.assertRaises(ValueError):
            self.add_0 / self.add_1

        with self.assertRaises(ValueError):
            self.add_0 / self.mult_1

        with self.assertRaises(ValueError):
            self.mult_1 / self.add_0

    def test_div(self):
        mult_0 = self.mult_1
        mult_1 = self.mult_5

        mult =   mult_0/mult_1

        self.assertEqual(mult.product, True)
        self.assertEqual(mult.shared_default, max(mult_0.shared_default, mult_1.shared_default))
        self.assertEqual(mult.numerical, 1.0)
        self.assertEqual(mult.symbolic, {'mult_1': 1, 'mult_5': -1})
        self.assertEqual(mult.defaults, {'mult_5': 5, 'mult_1': 1})

    def test_eval(self):
        ## using default values
        self.assertEqual(self.mult_1.eval(use_shared_default=False, feed_dict=None),1)
        self.assertEqual(self.add_5.eval(use_shared_default=False, feed_dict=None),5)

        ## using shared defaults
        self.assertEqual(self.mult_1.eval(use_shared_default=True, feed_dict=None),1)
        self.assertEqual(self.add_5.eval(use_shared_default=True, feed_dict=None),5)

        ## using feed dictionary
        self.assertEqual(self.mult_1.eval(use_shared_default=False, feed_dict={'mult_1':2}),2)
        self.assertEqual(self.add_5.eval(use_shared_default=False, feed_dict={'add_5':7}),7)

        ## the following lines are not strictly speaking a unit test and could be classified as integration test
        # Test addition
        addition = self.add_5+self.add_6
        self.assertEqual(addition.eval(use_shared_default=False), 11)
        self.assertEqual(addition.eval(use_shared_default=True), 12)
        self.assertEqual(addition.eval(use_shared_default=False, feed_dict={'add_5':4}), 10)
        self.assertEqual(addition.eval(use_shared_default=False, feed_dict={'add_5':4, 'add_6':5}), 9)

        # Test addition
        addition = self.add_5+6
        self.assertEqual(addition.eval(use_shared_default=False), 11)
        self.assertEqual(addition.eval(use_shared_default=True), 11)
        self.assertEqual(addition.eval(use_shared_default=False, feed_dict={'add_5':4}), 10)

        # Test addition
        addition = 6 + self.add_5
        self.assertEqual(addition.eval(use_shared_default=False), 11)
        self.assertEqual(addition.eval(use_shared_default=True), 11)
        self.assertEqual(addition.eval(use_shared_default=False, feed_dict={'add_5':4}), 10)

        # Test Multiplication symnum * symnum
        mult = self.mult_6*self.mult_5
        self.assertEqual(mult.eval(use_shared_default=False), 30)
        self.assertEqual(mult.eval(use_shared_default=True), 36)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_5':4}), 24)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_5':4, 'mult_6':5}), 20)

        # Test Multiplication constant * symnum
        mult = 3*self.mult_5
        self.assertEqual(mult.eval(use_shared_default=False), 15)
        self.assertEqual(mult.eval(use_shared_default=True), 15)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_5':4}), 12)

        # Test Multiplication symnum * constant
        mult = self.mult_5*3
        self.assertEqual(mult.eval(use_shared_default=False), 15)
        self.assertEqual(mult.eval(use_shared_default=True), 15)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_5':4}), 12)

        # Test division symnum/symnum
        mult = self.mult_1/self.mult_5
        self.assertEqual(mult.eval(use_shared_default=False), 0.2)
        self.assertEqual(mult.eval(use_shared_default=True), 1.0)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_5':4}), 0.25)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_5':4, 'mult_1':2}), 0.5)

        # Test division symnum/constant
        mult = self.mult_1/5
        self.assertEqual(mult.eval(use_shared_default=False), 0.2)
        self.assertEqual(mult.eval(use_shared_default=True), 0.2)
        self.assertEqual(mult.eval(use_shared_default=False, feed_dict={'mult_1':2}), 0.4)

    def test_str(self):
        """ Tests the conversion of SymNum to string."""

        self.assertEqual(self.mult_0.__str__(), '1.0 * mult_0**1')
        self.assertEqual(self.add_0.__str__(), '0.0 + add_0*1')
        self.assertEqual(str(self.mult_1/self.mult_5), '1.0 * mult_1**1 * mult_5**-1')
        self.assertEqual(str(self.mult_1/self.mult_5/2), '0.5 * mult_1**1 * mult_5**-1')
        self.assertEqual(str(self.mult_1*self.mult_5), '1.0 * mult_1**1 * mult_5**1')
        self.assertEqual(str(3*self.mult_1*self.mult_5), '3.0 * mult_1**1 * mult_5**1')
        self.assertEqual(str(self.add_5 + self.add_6), '0.0 + add_5*1 + add_6*1')
        self.assertEqual(str(2+self.add_5 + self.add_6), '2.0 + add_5*1 + add_6*1')

class TestTestbench(unittest.TestCase):

    def setUp(self):
        self.tb_empty = Testbench(Network())

        ff_net = Network()
        ff_net.add_node('a')
        ff_net.add_node('b')
        ff_net.add_node('c')

        ff_net.add_edge(Edge('a', 'b', phase=0.5, attenuation=0.8, delay=2))
        ff_net.add_edge(Edge('b', 'c', phase=-5, attenuation=1.5, delay=-1))
        self.ff_net = ff_net

        self.tb_ff = Testbench(network=self.ff_net,timestep=1)

        self.x1 = np.array([0,1,6,7])
        self.x2 = np.array([0,2,3])
        self.t1 = np.array([0,2,5,7,9])
        self.t2 = np.array([0,2,5,12])


    def test_set_feed_dict(self):
        self.tb_empty.set_feed_dict({'a':2,'b':3})
        self.assertEqual(self.tb_empty.feed_dict,{'a':2,'b':3})

    def test_add_input(self):

        x1 = self.x1
        x2 = self.x2
        t1 = self.t1
        t2 = self.t2

        t_short = np.array([0,2,5,7])

        with self.assertRaises(ValueError):
            self.tb_ff.add_input_sequence('a',x=x1, t=t_short)

        with self.assertRaises(ValueError):
            self.tb_ff.add_input_sequence('d',x=x1, t=t1)

        self.tb_ff.add_input_sequence('a',x=x1, t=t1)
        self.assertEqual(self.tb_ff.input_nodes, ['a'])
        self.assertEqual(self.tb_ff.model.inputs, [(1.0,0.0,0.0,'a')])
        self.assertEqual(self.tb_ff.input_x, [x1] )
        self.assertEqual(self.tb_ff.input_t, [t1] )

        self.tb_ff.add_input_sequence('b',x=x2, t=t2)
        self.assertEqual(self.tb_ff.input_nodes, ['a','b'])
        self.assertEqual(self.tb_ff.model.inputs, [(1.0,0.0,0.0,'a'),(1.0,0.0,0.0,'b')])
        self.assertEqual(self.tb_ff.input_x, [x1, x2] )
        self.assertEqual(self.tb_ff.input_t, [t1, t2] )

        with self.assertRaises(OverflowError):
            self.tb_ff.add_input_sequence('a',x=x1, t=t1)

    def test_extract_min_max_signal_time(self):
        self.tb_ff.add_input_sequence('a',x=self.x1, t=self.t1)
        self.tb_ff.add_input_sequence('b',x=self.x2, t=self.t2)
        self.tb_ff._extract_min_max_signal_time()
        self.assertEqual(self.tb_ff.t0, 0)
        self.assertEqual(self.tb_ff.t1, 12)

    def test_convert_signals_to_timestep(self):
        self.tb_ff.t0 = 0
        self.tb_ff.t1 = 2

        t_sampled, x_sampled = self.tb_ff._convert_signal_to_timestep(x=self.x1, t=self.t1, timestep=0.5)
        self.assertEqual(np.all(t_sampled==[0,0.5,1,1.5,2]), True)
        self.assertEqual(np.all(x_sampled==np.array([0,0,0,0,1], dtype=np.complex)),True )

    def test_interpolate_constant(self):
        # Case 1: no padding required, critical matching points (e.g. step = 0.5; change at 1)
        expected_result = np.array([2,2,4,4,4,4,1,1,1,1,1],dtype=np.complex)
        self.assertEqual(np.all(self.tb_empty._interpolate_constant(x=np.linspace(0,5,11), xp=[0, 1, 3, 6], yp=[2,4,1])==expected_result), True)

        # Case 2:  right boundary case
        expected_result = np.array([2,2,4,4,4,4,1,1,1,1,0],dtype=np.complex)
        self.assertEqual(np.all(self.tb_empty._interpolate_constant(x=np.linspace(0,5,11), xp=[0, 1, 3, 5], yp=[2,4,1])==expected_result), True)

        # Case 3:  left boundary case
        expected_result = np.array([0,2,4,4,4,4,1,1,1,1,1],dtype=np.complex)
        self.assertEqual(np.all(self.tb_empty._interpolate_constant(x=np.linspace(0,5,11), xp=[0.5, 1, 3, 6], yp=[2,4,1])==expected_result), True)

        # Case 4:  right + left boundary case
        expected_result = np.array([0,2,4,4,4,4,1,1,1,1,0],dtype=np.complex)
        self.assertEqual(np.all(self.tb_empty._interpolate_constant(x=np.linspace(0,5,11), xp=[0.5, 1, 3, 5], yp=[2,4,1])==expected_result), True)

        # Case 5:  non-matching timesteps with left boundary case
        expected_result = np.array([0,0, 2, 4,4,4,4,1,1,1,1,1],dtype=np.complex)
        self.assertEqual(np.all(self.tb_empty._interpolate_constant(x=np.linspace(0,5,12), xp=[0.5, 1, 3, 6], yp=[2,4,1])==expected_result), True)