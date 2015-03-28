
# Mnemonic access indeces
obsHandPose, obsGrip, obsTrigger, obsContacts = range(4)

def reactiveApproach(robot, graspPose, gripDes, offset = 0.05, tries = 10):
    if tries == 0:
        print 'reactiveApproach failed'
        return None

    (obs, traj) = robot.tryGrasp(graspPose, offset = offset)
    print 'obs after tryGrasp', obs
    if reactBoth(obs):
        if abs(obs[obsGrip] - gripDes) < 0.03:
            print 'holding'
            return obs
        else:
            print 'opening', obs[obsGrip], 'did not match', gripDes
            gripOpen(robot)
    if reactLeft(obs):
        print 'reactLeft'
        robot.place(rob.Conf(robot.conf.basePose, \
                             displacePose(graspPose, -offset, 0)))
        newGraspPose = displacePose(graspPose, 0.0, 0.01)
        reactiveApproach(robot, newGraspPose, gripDes, offset, tries-1)
    elif reactRight(obs):
        print 'reactRight'
        robot.place(rob.Conf(robot.conf.basePose, \
                             displacePose(graspPose, -offset, 0)))
        newGraspPose = displacePose(graspPose, 0.0, -0.01)
        reactiveApproach(robot, newGraspPose, gripDes, offset, tries-1)
    else:
        print 'reactiveApproach confused'
        return None

def displacePose(pose, dx, dy):
    pose = pose.pose()
    newPoint = pose.point() + pose.applyToPoint(util.Point(dx, dy, 0, w=0))
    return newPoint.pose(pose.theta)
def reactBoth(obs):
    return obs[obsContacts][1] and obs[obsContacts][3] 
def reactRight(obs):
    return obs[obsTrigger] in ('R_tip', 'R_pad') \
           or obs[obsContacts][2] or obs[obsContacts][3]
def reactLeft(obs):
    return obs[obsTrigger] in ('L_tip', 'L_pad') \
           or obs[obsContacts][0] or obs[obsContacts][1]
def gripOpen(robot):
    robot.kinPlace(robot.kinConf(util.Pose(0,0,0,0), 8*[0.0], 0.08),
                   ignoreBase = True, ignoreArm = True)

    def tryGrasp(self, graspPose, offset = glob.approachOffsetX,
                 nsteps = 3, verbose = False):

        startConf = self.conf
        graspConf = rob.Conf(startConf.basePose, graspPose)
        approachPose = graspPose.compose(util.Pose(-offset,0,0,0))
        approachConf = rob.Conf(startConf.basePose, approachPose)

        self.place(approachConf)     # move to approach conf
        print 'At approach conf'
        pr2GoToConfGuarded(resetForce = True)
        trigger = self.guardedMove(approachConf, graspConf, n=nsteps)
        print 'trigger', trigger
        # raw_input('At reach conf')
        if trigger in ('Acc', 'goal'):
            result = self.compliantClose(0.005)
            print 'compliantClose result', result
            if result == 'LR_pad':
                contacts = [False, True, False, True]
            elif result == 'none':
                contacts = 4*[False]
            else:
                raw_input('Unexpected result from compliantClose')
                contacts = 4*[False]
        else:
            contacts = 4*[False]

        obs = (self.conf.handPose, self.conf.grip, trigger, contacts)
        
        return obs, (approachConf, self.conf)

def compliantClose(step = 0.01, n = 1):
    if n > 5:
        (result, cnfOut) = pr2GoToConf('close')
        return result
    if self.useROS and not glob.disableRobot:
        print 'compliantClose step=', step
        # raw_input('about to call guarded close')
        (result, kc) = pr2GoToConfGuarded(grip = 'either')
        print 'result', result
        print 'kc', kc
        # raw_input('finished guarded close')
        if kc:
            self.kinPlace(kc, moveRealRobot = False)
        else:
            raw_input('guarded close failed')
            return None

        # could displace to find contact with the other finger
        # instead of repeatedly closing.
        if result == 'LR_pad':
            return result
        elif result in ('L_pad', 'R_pad'):
            off = step if result == 'L_pad' else -step
            hand = self.conf.handPose.pose()
            hp = hand.applyToPoint(util.Point(0, off, 0)).pose(hand.theta)
            nkc = self.invKin(rob.Conf(self.conf.basePose, hp),
                              collisionAware = True)
            # raw_input('about to adjust hand')
            self.kinPlace(nkc, ignoreBase = True, ignoreGrip = True)
            # raw_input('finished adjusting hand, ready to call compliantClose again')
            return self.compliantClose(0.9*step, n = n+1)
        elif result == 'none':
            return result
        else:
            raw_input('Bad result in compliantClose: %s'%str(result))
            return result
    else:
        raw_input('compliantClose does not work in simulation')

def pr2GoToConfGuarded(cnfIn, speedFactor = 0.25):

    try:
        gotoConf = rospy.ServiceProxy('pr2_goto_configuration',
                                      pr2_hpn.srv.GoToConf)
        conf = pr2_hpn.msg.Conf()
        conf.arm = 'r' if glob.right else 'l'
        conf.joints = map(float, joints)
        conf.base = conf.torso = conf.grip = conf.head = []

        if resetForce:
            operation = 'resetForce'
        elif grip:
            if grip == 'either': operation = 'closeGuarded'
            elif grip == 'grab': operation = 'grab'
            else: operation = 'close'
        else:
            operation = 'moveGuarded'

        resp = gotoConf(operation, conf, speedFactor)
        print 'response', resp
        if resp:
            c = resp.resultConf
            joints = [util.fixAnglePlusMinusPi(a) for a in c.joints]
            kc = rob.KConf(util.Pose(c.base[0], c.base[1], 0, c.base[2]),
                           c.torso + tuple(joints), c.grip[0])
            return resp.result, kc
        else:
            return None, None
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

    def guardedMove(self, conf1, conf2, n=4):
        name = 'guardedMove'
        hand1 = conf1.handPose.pose()
        hand2 = conf2.handPose.compose(util.Pose(glob.pickOvershoot,0,0,0)).pose()
        assert abs(util.fixAnglePlusMinusPi(hand1.theta - hand2.theta)) < 0.001
        p1 = hand1.point()
        p2 = hand2.point()
        delta = (p2 - p1).scale(1.0/n)
        for i in range(1, n+1):
            hand = (p1 + delta.scale(i)).pose(hand1.theta)
            kinConf = self.invKin(rob.Conf(conf1.basePose, hand, conf1.grip),
                                  collisionAware = True)
            
            if not kinConf:
                if self.world: self.draw(self.world.window, color = 'purple')
                print name, 'Step', i
                print 'Failute of inverse kinematics'
                raw_input('Infeasible step in guarded move; enter to continue')
                continue
            
            status, kc = self.kinPlaceGuarded(kinConf)
            
            print i, 'Moving to', hand, 'result is', status
            if debug(name) and self.realRobot:
                colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
                if self.world: self.draw(self.world.window, color = colors[i%6])
                print name, 'Step', i
                print kinConf
                raw_input('go?')

            if not status == 'goal': return status
        return 'goal'

    def kinPlaceGuarded(self, kinConf, moveRealRobot = True):
        if self.realRobot and moveRealRobot:
            # Move real robot or environment simulated robot
            if self.useROS:
                # Move actual robot
                (base, zth, gopen) = (kinConf.basePose, kinConf.angles,
                                      kinConf.grip)
                status, kc = pr2GoToConfGuarded(joints=zth[1:])
                if kc:                  # actual KConf, update model
                    self.kinPlace(kc, moveRealRobot = False)
                else:
                    raw_input('GoToConfGuarded failed')
        return status, self.kinConf
