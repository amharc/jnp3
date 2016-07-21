{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE DataKinds #-}
module TH where

import Data.Functor.Classes
import Data.Functor.Identity
import Language.Haskell.TH
import Linear
import Model
import GHC.TypeLits
import GHC.Prim

class IsScalarFun (name :: Symbol) where
    runScalarFun :: Floating a => Proxy# name -> a -> a

makeScalarFun :: ExpQ -> String -> Q [Dec]
makeScalarFun fun name = [d|
        instance IsScalarFun $(nameLit) where
            runScalarFun _ = $(fun)
            {-# INLINE runScalarFun #-}
    |]
  where
    nameLit = litT $ strTyLit name
