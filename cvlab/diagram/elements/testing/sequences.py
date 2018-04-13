# -*- coding: utf-8 -*-

class SeqTest(SequenceToDataElement, FunctionGuiElement):
    name = "Sequence processing test"
    comment = ""

    def get_processing_units(self, inputs, parameters):
        outputs = {"data": Data(), "seq": Sequence([Data() for _ in xrange(4)]), "float": Data()}
        units = [ProcessingUnit(self, inputs, parameters, outputs)]
        return units, outputs

    def get_attributes(self):
        return [Input("i1"), Input("i2")], [Output("data"), Output("seq"), Output("float")], [IntParameter("x")]

    def process_inputs(self, inputs, outputs, parameters):
        print "SeqTest.process_inputs. inputs:", inputs
        outputs["data"] = Data(np.random.rand(10, 10))
        outputs["seq"] = Sequence([Data(np.random.rand(10, 10)) for _ in xrange(4)])
        outputs["float"] = Data(np.random.rand())
