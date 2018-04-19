import socket
import nmap

def inMyNetwork(hosts='192.168.1.0/24', arguments='-n -sP -PE -PA21,23,80,3389'):
    """
    generate list of devices and there IP addresses who connected with you in the same Network
    :param str() hosts: string for hosts as nmap use it 'scanme.nmap.org' or '198.116.0-255.1-127' or '216.163.128.20/20'
    :param arguments: string of arguments for nmap '-sU -sX -sC'
    """
    nm = nmap.PortScanner()
    nm.scan(hosts=hosts,arguments=arguments)
    hosts_list = [(x, socket.gethostbyaddr(x)[0]) for x in nm.all_hosts() if socket.gethostbyaddr(x)]
    for host, name in hosts_list:
        print('{0}:{1} '.format(name, host))

def ClusterGen(workers,pss,port=2222):
    """
    function to generate list of IP addresses for workers and parameters servers from Devices name
    :param list[str] workers: list of workers Devices name
    :param list[str] pss: list of parameter servers Devices name
    :param int() port: port number
    """
    l = []
    d = []
    for worker in workers:
        try:
            l.append(socket.gethostbyname(worker)+":"+str(port))
        except:
            print("worker {} Hors ligne ".format(worker))
    for ps in pss:
        try:
            d.append(socket.gethostbyname(ps)+":"+str(port))
        except:
            print("ps {} Hors ligne".format(ps))
    return {'worker':l,'ps':d}




if __name__ == '__main__':

    workers = ['DESKTOP-07HFBQN','fouzi-pc']
    pss = ['DELL-MINI']

    print(ClusterGen(workers,pss))

    inMyNetwork()





