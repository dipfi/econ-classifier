import configparser

config = configparser.ConfigParser()
if (os.getcwd()=='/home/guillotm')| (os.getcwd()=='/home/malka'):
    config.read(os.getcwd()+'/Dropbox/postdoc_eth/projets/firm-registry-ch/code/config_mg.ini')
if os.getcwd()=='/cluster/work/lawecon/Projects/Ash_Guillot/firm-registry-ch/code/pre-2001':
    config.read('/cluster/work/lawecon/Projects/Ash_Guillot/firm-registry-ch/code/config.ini')
if os.getcwd()=='/Users/annastuenzi/Dropbox (squadrat-architekten)/firm-registry-ch/code/pre-2001':
    config.read('/cluster/work/lawecon/Projects/Ash_Guillot/firm-registry-ch/code/config.ini')

data_path=config['PATH']['data_path']
code_path=config['PATH']['projet'] +'/code'