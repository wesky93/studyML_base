import unittest
from game import Games



class MyTestCase( unittest.TestCase ) :

    def make_test( self ) :
        result = True
        try:
            game = Games( )
            game.make_game()
            print(game.states)
            print(game.rewards)
        except Exception as e:
            result = False
        self.assertEqual( True, result )

    def setate_test(self):
        game = Games( )
        game.make_game( )
        print( game.states )


if __name__ == '__main__' :
    unittest.main( )
