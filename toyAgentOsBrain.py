from osbrain import run_agent
from osbrain import run_nameserver
from osbrain import Agent
import osbrain

#We will work with messages serialized in json
osbrain.config['SERIALIZER'] = 'json'

#Simple class for a merchant
class Merchant(Agent):
    budget = 100
    preference = None

    def on_init(self):
        pass

    def on_new_msg(self, msg):
        self.log_info(msg)


#Simple class for an operator
class Operator(Agent):

    def on_init(self):
        # we create a communication of type publisher name "publish channel"
        self.bind('PUB', alias='publish_channel')

    def send_new_product(self):
        # we send a new product to the merchants subscribed
        self.send('publish_channel', {"product number": 1, "product type" : "H", "price": 30})


if __name__ == '__main__':

    # init
    ns = run_nameserver()
    Operator = run_agent('Operator', base=Operator)
    Merchant = run_agent('Merchant', base=Merchant)
    
    # we connect the merchant to the operator channel publish channel
    Merchant.connect(Operator.addr('publish_channel'), handler='on_new_msg')

    # Log a message
    for alias in ns.agents():
        print(alias)

    Operator.send_new_product()

    ns.shutdown()