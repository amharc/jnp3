{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}
module Model where

import Linear

class (Functor (Input m), Functor (Output m)) => Model m where
    type Input m :: * -> *
    type Output m :: * -> *
    runModel :: Floating a => Input m a -> m a -> Output m a

-- | An auxiliary class computing the L2 norm of the non-bias coefficients.
class HasL2 m where
    getL2 :: Num a => m a -> a
