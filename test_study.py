import unittest
import numpy as np
from study import windows, metrics, softmax, select_threshold, selective, WINDOW, PURGE

class ProtocolTests(unittest.TestCase):
    def fixture(self):
        a = np.tile(np.arange(1000,dtype='float32'),(45,1))
        a[-5:] = 2
        return a
    def test_causal_window_and_endpoint_label(self):
        x,y,ends = windows(self.fixture())
        np.testing.assert_array_equal(x[:,-1,0],ends)
        np.testing.assert_array_equal(x[:,0,0],ends-WINDOW+1)
        self.assertTrue((y==1).all())
    def test_purged_split_excludes_future_labels(self):
        a=self.fixture()
        _,_,left=windows(a,0,500-PURGE)
        _,_,right=windows(a,500+PURGE)
        self.assertLess(left[-1]+10,right[0]-WINDOW+1)
    def test_invalid_labels_rejected(self):
        a=self.fixture(); a[-5]=0
        with self.assertRaises(AssertionError): windows(a)
    def test_nonfinite_features_rejected(self):
        a=self.fixture(); a[0,:]=np.nan
        with self.assertRaises(AssertionError): windows(a)
    def test_temperature_preserves_classes(self):
        z=np.array([[1,2,3],[4,2,1]],dtype=float)
        np.testing.assert_array_equal(softmax(z).argmax(1),softmax(z,3).argmax(1))
        np.testing.assert_allclose(softmax(z).sum(1),1)
    def test_brier_perfect(self):
        m=metrics(np.array([0,1,2]),np.eye(3))
        self.assertEqual(m['brier'],0)
        self.assertEqual(m['macro_f1'],1)
    def test_threshold_uses_supplied_calibration_confidences(self):
        p=np.array([[.6,.2,.2],[.8,.1,.1]])
        self.assertAlmostEqual(select_threshold(p),.7)
    def test_empty_selection_is_explicit(self):
        m=selective(np.array([0]),np.array([[.4,.3,.3]]),.9)
        self.assertIsNone(m['accuracy_retained'])
        self.assertEqual(m['coverage'],0)

if __name__=='__main__': unittest.main()
