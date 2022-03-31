import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run spDANMF example.")
    parser.set_defaults(layers=[256,128,42])
    parser.set_defaults(eta=1.05)
    parser.set_defaults(lamb=0.0001)
    parser.set_defaults(dataSetName='email-Eu-core')
    parser.set_defaults(bigIterations=20)
    parser.set_defaults(iterations=20)
    parser.set_defaults(pre_iterations=100)
    parser.set_defaults(classNum=42)
    parser.set_defaults(calculate_loss=False)
    return parser.parse_args()




