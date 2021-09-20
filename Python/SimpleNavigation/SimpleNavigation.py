nodesList=((31.251764, 34.791202, 3), (31.252042, 34.786731, 2),
(31.251679, 34.793161, 1), (31.250381, 34.784395, 2),
(31.247839, 34.783515, 2), (31.246378, 34.782915, 1),
(31.245723, 34.786335, 3), (31.246487, 34.787483, 3),
(31.245117, 34.789172, 1), (31.248612, 34.789633, 2),
(31.250666, 34.790588, 1))
#(id1,id2,oneDirection) - the ids are actually the indexes in the nodes tuple (starting from 0 of course)
sectionsList=((0,1,1),(0,2,1),(1,3,0),(3,4,0),(4,5,0),(5,6,0),(7,6,1),(6,8,0),(7,9,0),(9,10,1),(10,0,0))
import random
class Node:
    ID=0#to put an different id to each Node that made
    def __init__(self,geoX,geoY,delayTime=0):
        '''init function that get self,geoX,geoY and delayTime(will be 0 if not sent) and make a new Node'''
        self.GeoX=geoX
        self.GeoY=geoY
        self.id=Node.ID
        Node.ID=Node.ID+1
        self.delayTime=delayTime
    def str(self):
        '''function that return a node form as a node # id (x,y)'''
        return 'node #{0} ({1},{2})'.format(self.id,self.GeoX,self.GeoY)
    def __eq__(self,other):
        '''function that get self and other and compares between them,return true if they have the same values'''
        if isinstance(other,Node):
            return self.GeoX==other.GeoX and self.GeoY==other.GeoY
        else:
            return 'the second object is not Node object!'
class Section():
    def __init__(self,n1,n2):
        if isinstance(n1,Node) and isinstance(n2,Node):
            self.n1=n1
            self.n2=n2
            temp=1000*(((n2.GeoX-n1.GeoX)**2)+((n2.GeoY-n1.GeoY)**2)**0.5)
            if temp%0.001>0:
                temp=((temp-temp%0.001)+0.001)
            else:
                temp=(temp-temp%0.001)
            self.length=temp
            self.avgSpeed=random.randrange(10,80)
    def str(self):
        return 'from '+self.n1.str()+' to '+self.n2.str()
    def calcAvgTime(self,userSpeed):
        return (self.length/max(userSpeed,self.avgSpeed))*60
class DirectionSection(Section):
    def __init__(self,nFrom,nTo,oneDirection):
        self.Direction=oneDirection
        super().__init__(nFrom,nTo)
class NoRouteException(Exception):
    def __init__(self,msg):
        self.msg=msg
class Navigation():
    sections=[]
    nodes=[]
    def __init__(self,transport,nodesList,sectionsList):
        for i in range(0,len(nodesList)):
            Navigation.nodes.append(Node(nodesList[i][0],nodesList[i][1],nodesList[i][2]))
        for i in range(0,len(sectionsList)):
            if sectionsList[i][2]==1:
                Navigation.sections.append(DirectionSection(Navigation.nodes[sectionsList[i][0]],Navigation.nodes[sectionsList[i][1]],True))
            else:
                Navigation.sections.append(DirectionSection(Navigation.nodes[sectionsList[i][0]],Navigation.nodes[sectionsList[i][1]],False))
            if transport=='walk':
                self.userSpeed=6
            elif transport=='car':
                self.userSpeed=random.randrange(10,70)
            elif transport=='bicycle':
                self.userSpeed=random.randrange(8,16)
    def printAllNodes(self):
        for n in Navigation.nodes:
            print(n.str()+' delay:{0}'.format(n.delayTime))
    def navigate(self,id1,id2):
        def calcRouteTime(start,end):
            def searchSection(id1,id2):
                for s in Navigation.sections:
                    
                    if ((s.n1).id==id1 and (s.n2).id==id2) or ((s.n2).id==id1 and (s.n1).id==id2 and s.Direction==False):
                        return s
                raise NoRouteException('No such route exists')
        
            try:
                if start.id<end.id:
                    tempobj=searchSection(start.id,start.id+1)
                    if tempobj.n1==start:
                        temp=calcRouteTime(tempobj.n2,end)
                        delay=(tempobj.n2).delayTime
                    else:
                        temp=calcRouteTime(tempobj.n1,end)
                        delay=(tempobj.n1).delayTime
                    if temp==-1:#cuz calcRouteTime is recursive,we shall check everytime if returned -1 so we don't calc the time.
                        return -1
                    return temp+tempobj.calcAvgTime(self.userSpeed)+delay
                return 0
            except NoRouteException as e:
                print(e.msg)
                return -1
                
        if id1>Navigation.nodes[len(Navigation.nodes)-1].id or id1<Navigation.nodes[0].id or id2>Navigation.nodes[len(Navigation.nodes)-1].id or id2<Navigation.nodes[0].id or id1>=id2:
            return 'please make sure you enter correct ids'
        else:
            check=calcRouteTime(Navigation.nodes[id1],Navigation.nodes[id2])
            if check!=-1:
                check=format(check,'.3f')
                return 'navigation time: {0} min'.format(check)
            else:
                return ''#if we would not return empty str here, it will return None  
    

#driver - use as is
wise=Navigation(input('choose transportation: walk/car/bicycle: '),nodesList,sectionsList)
wise.printAllNodes()
print('navigate')            
startId=int(input('from:'))
endId=int(input('to:'))
print(wise.navigate(startId,endId))       
    
            
