import unittest

# 단순하게 가중치만 변경함으로서 and,or,nand 를 구현함

def AND( v1, v2 ) :
    w1,w2,theta = 0.5,0.5,0.7

    func = v1*w1+v2*w2
    if func <= theta:
        return 0
    elif func > theta:
        return 1



def OR( v1, v2 ) :
    w1,w2,theta = 0.5,0.5,0.2
    func = v1 * w1 + v2 * w2
    if func <= theta:
        return 0
    elif func > theta:
        return 1



def NAND( v1, v2 ) :
    w1, w2, theta = -0.5, -0.5, -0.7

    func = v1 * w1 + v2 * w2
    if func <= theta :
        return 0
    elif func > theta :
        return 1

Q = [ (0, 0), (0, 1), (1, 0), (1, 1) ]

class MyTestCase( unittest.TestCase ) :

    def test_and( self ) :
        A = [ 0,0,0,1]
        for q, a in zip( Q, A ) :
            self.assertEqual( a, AND( q[ 0 ], q[ 1 ] ) )

    def test_or( self ) :
        A = [0,1,1,1 ]
        for q, a in zip( Q, A ) :
            self.assertEqual( a, OR( q[ 0 ], q[ 1 ] ) )

    def test_nand( self ) :
        A = [ 1,1,1,0]
        for q, a in zip( Q, A ) :
            self.assertEqual( a, NAND( q[ 0 ], q[ 1 ] ) )


if __name__ == '__main__' :
    unittest.main( )
